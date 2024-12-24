use quote::{format_ident, quote};
use syn::{parse_macro_input, Data, DeriveInput, Fields, Type};

#[proc_macro_derive(StwoIterable)]
pub fn derive_stwo_iterable(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = &input.ident;
    let input = match input.data {
        Data::Struct(data_struct) => data_struct,
        _ => panic!("Expected struct"),
    };

    let fields = match input.fields {
        Fields::Named(fields) => fields.named,
        _ => panic!("Expected named fields"),
    };

    let mut non_array_field_names = vec![];
    let mut non_array_inner_array_sizes = vec![];

    let mut array_field_names = vec![];
    let mut outer_array_sizes = vec![];
    let mut inner_array_sizes = vec![];

    let mut first_field_vector = None;

    for field in fields {
        match field.ty {
            Type::Array(outer_array) => {
                if first_field_vector.is_none() {
                    let ident = field.ident.clone().unwrap();
                    first_field_vector = Some(quote! { #ident[0] });
                }
                array_field_names.push(field.ident.unwrap());
                outer_array_sizes.push(outer_array.len.clone());

                if let Type::Path(type_path) = *outer_array.elem {
                    if let Some(last_segment) = type_path.path.segments.last() {
                        if last_segment.ident != "Vec" {
                            panic!("Expected Vec type");
                        }
                        if let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments {
                            if let Some(syn::GenericArgument::Type(Type::Array(array))) =
                                args.args.first()
                            {
                                inner_array_sizes.push(array.len.clone());
                            }
                        }
                    }
                }
            }
            Type::Path(type_path) => {
                if first_field_vector.is_none() {
                    let ident = field.ident.clone().unwrap();
                    first_field_vector = Some(quote! { #ident });
                }
                non_array_field_names.push(field.ident.unwrap());
                if let Some(last_segment) = type_path.path.segments.last() {
                    if last_segment.ident != "Vec" {
                        panic!("Expected Vec type");
                    }
                    if let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments {
                        if let Some(syn::GenericArgument::Type(Type::Array(array))) =
                            args.args.first()
                        {
                            non_array_inner_array_sizes.push(array.len.clone());
                        }
                    }
                }
            }
            _ => panic!("Expected vector or array of vectors"),
        }
    }

    let (non_array_field_names_head_prefix, non_array_field_names_tail_prefix): (Vec<_>, Vec<_>) =
        non_array_field_names
            .iter()
            .map(|field_name| {
                (
                    format_ident!("{}_head", field_name),
                    format_ident!("{}_tail", field_name),
                )
            })
            .unzip();
    let (array_field_names_head_prefix, array_field_names_tail_prefix): (Vec<_>, Vec<_>) =
        array_field_names
            .iter()
            .map(|field_name| {
                (
                    format_ident!("{}_head", field_name),
                    format_ident!("{}_tail", field_name),
                )
            })
            .unzip();

    let mut_chunk_name = format_ident!("{}MutChunk", struct_name);
    let iter_mut_name = format_ident!("{}IterMut", struct_name);
    let row_producer_name = format_ident!("{}RowProducer", struct_name);
    let par_iter_mut_name = format_ident!("Par{}IterMut", struct_name);

    let expansions = quote! {
        use stwo_prover::core::backend::simd::m31::PackedM31;
        use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
        use stwo_prover::core::backend::simd::m31::N_LANES;
        use rayon::prelude::*;

        impl #struct_name {
            /// # Safety
            /// The caller must ensure that the trace is populated before being used.
            pub unsafe fn uninitialized(log_size: u32) -> Self {
                let length = 1 << log_size;
                let n_simd_elems = length / N_LANES;
                #(
                    let mut #non_array_field_names = Vec::with_capacity(n_simd_elems);
                    #non_array_field_names.set_len(n_simd_elems);
                )*
                #(
                    let #array_field_names = [(); #outer_array_sizes].map(|_| {
                        let mut vec = Vec::with_capacity(n_simd_elems);
                        vec.set_len(n_simd_elems);
                        vec
                    });
                )*
                Self { #(#array_field_names),*,#(#non_array_field_names),* }
            }

            pub fn iter_mut(&mut self) -> #iter_mut_name<'_> {
                #iter_mut_name::new(
                    #(&mut self.#non_array_field_names),*,
                    #(&mut self.#array_field_names),*
                )
            }

            pub fn par_iter_mut(&mut self) -> #par_iter_mut_name<'_> {
                #par_iter_mut_name::new(
                    #(&mut self.#non_array_field_names),*,
                    #(&mut self.#array_field_names),*
                )
            }
        }

        pub struct #mut_chunk_name<'trace> {
            #(#non_array_field_names: &'trace mut [PackedM31; #non_array_inner_array_sizes],)*,
            #(#array_field_names: [&'trace mut [[PackedM31; #inner_array_sizes]]; #outer_array_sizes],)*
        }

        pub struct #iter_mut_name<'trace> {
            #(#non_array_field_names: *mut [[PackedM31; #non_array_inner_array_sizes]],)*
            #(#array_field_names: [*mut [[PackedM31; #inner_array_sizes]]; #outer_array_sizes],)*
            phantom: std::marker::PhantomData<&'trace ()>,
        }

        impl<'trace> #iter_mut_name<'trace> {
            pub fn new(
                #(#non_array_field_names: &'trace mut [[PackedM31; #non_array_inner_array_sizes]],)*
                #(#array_field_names: [&'trace mut [[PackedM31; #inner_array_sizes]]; #outer_array_sizes],)*
            ) -> Self {
                Self {
                    #(#non_array_field_names: #non_array_field_names.as_mut_ptr(),)*
                    #(#array_field_names: #array_field_names.map(|v| v.as_mut_ptr()),)*
                    phantom: std::marker::PhantomData,
                }
            }
        }

        impl<'trace> Iterator for #iter_mut_name<'trace> {
            type Item = #mut_chunk_name<'trace>;
            fn next(&mut self) -> Option<Self::Item> {
                unsafe {
                    if self.#first_field_vector.is_empty() {
                        return None;
                    }
                    #(
                        let (#non_array_field_names_head_prefix, #non_array_field_names_tail_prefix) = self.#non_array_field_names.split_at_mut(1);
                        self.#non_array_field_names = #non_array_field_names_tail_prefix;
                    )*
                    #(
                        let #array_field_names_head_prefix = self.#array_field_names.iter_mut().map(|ptr| {
                            let (head, tail) = ptr.split_at_mut(1);
                            *ptr = tail;
                            &mut head[0]
                        });
                    )*
                    let item = #mut_chunk_name {
                        #(#non_array_field_names: #non_array_field_names_head_prefix,)*,
                        #(#array_field_names: #array_field_names_head_prefix,)*
                    };
                    Some(item)
                }
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.#first_field_vector.len();
                (len, Some(len))
            }
        }

        impl ExactSizeIterator for #iter_mut_name<'_> {}

        impl<'trace> DoubleEndedIterator for #iter_mut_name<'trace> {
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.#first_field_vector.is_empty() {
                    return None;
                }
                #(
                    let (#non_array_field_names_head_prefix, #non_array_field_names_tail_prefix)
                    = self.#non_array_field_names.split_at_mut(non_array_field_names.len() - 1);
                    self.#non_array_field_names = #non_array_field_names_head_prefix;
                )*
                #(
                    let #array_field_names_tail_prefix = self.#array_field_names.iter_mut().map(|ptr| {
                        let (head, tail) = ptr.split_at_mut(ptr.len() - 1);
                        *ptr = head;
                        &mut tail[0]
                    });
                )*
                let item = #mut_chunk_name {
                    #(#non_array_field_names: #non_array_field_names_tail_prefix,)*,
                    #(#array_field_names: #array_field_names_tail_prefix,)*
                };
                Some(item)
            }
        }

        pub struct #row_producer_name<'trace> {
            #(#non_array_field_names: &'trace mut [[PackedM31; #non_array_inner_array_sizes]],)*
        }

        impl<'trace> Producer for  #row_producer_name<'trace> {
            type Item = #mut_chunk_name<'trace>;
            type IntoIter = #iter_mut_name<'trace>;

            fn split_at(self, index: usize) -> (Self, Self) {
                #(
                    let (#non_array_field_names, #non_array_field_names_tail_prefix) = self.#non_array_field_names.split_at_mut(index);
                )*
                #(
                    let (#array_field_names, #array_field_names_tail_prefix) = self.#array_field_names.map(|v| v.as_mut_slice()).split_at_mut(index);
                )*
                (
                    #row_producer_name { #(#non_array_field_names),* ,#(#array_field_names),* },
                    #row_producer_name { #(#non_array_field_names_tail_prefix),* ,#(#array_field_names_tail_prefix),* },
                )

            }

            fn into_iter(self) -> Self::IntoIter {
                #iter_mut_name::new(#(self.#non_array_field_names),*,#(self.#array_field_names),*)
            }
        }

        pub struct #par_iter_mut_name<'trace> {
            #(
                #non_array_field_names: &'trace mut [[PackedM31; #non_array_inner_array_sizes]],
            )*,
            #(
                #array_field_names: [&'trace mut [[PackedM31; #inner_array_sizes]]; #outer_array_sizes],
            )*
        }

        impl<'trace> #par_iter_mut_name<'trace> {
            pub fn new(
                #(#non_array_field_names: &'trace mut [[PackedM31; #non_array_inner_array_sizes]],)*
                #(#array_field_names: [&'trace mut [[PackedM31; #inner_array_sizes]]; #outer_array_sizes],)*
            ) -> Self {
                Self { #(#non_array_field_names,)*, #(#array_field_names,)* }
            }
        }

        impl<'trace> ParallelIterator for #par_iter_mut_name<'trace> {
            type Item = #mut_chunk_name<'trace>;

            fn drive_unindexed<D>(self, consumer: D) -> D::Result
            where
                D: UnindexedConsumer<Self::Item>,
            {
                bridge(self, consumer)
            }

            fn opt_len(&self) -> Option<usize> {
                Some(self.len())
            }
        }

        impl IndexedParallelIterator for #par_iter_mut_name<'_> {
            fn len(&self) -> usize {
                self.#first_field_vector.len()
            }

            fn drive<D: Consumer<Self::Item>>(self, consumer: D) -> D::Result {
                bridge(self, consumer)
            }

            fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
                callback.callback(
                    #row_producer_name { 
                        #(#non_array_field_names: self.#non_array_field_names,)* 
                        #(#array_field_names: self.#array_field_names,)* 
                    }
                )
            }
        }
    };

    proc_macro::TokenStream::from(expansions)
}

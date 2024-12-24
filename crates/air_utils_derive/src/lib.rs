use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Data, DeriveInput, Expr, Fields, Ident, Type};

#[allow(dead_code)]
trait IterableField {
    fn name(&self) -> &Ident;
    fn r#type(&self) -> &Type;
    fn mut_slice_type(&self) -> TokenStream;
    fn mut_ptr_type(&self) -> TokenStream;
    fn uninitialized(&self) -> TokenStream;
    fn split_first(&self) -> TokenStream;
    fn split_last(&self) -> TokenStream;
    fn split_at(&self, index: Ident) -> TokenStream;
    fn as_mut_slice(&self) -> TokenStream;
    fn len(&self) -> TokenStream;
}

struct PlainVec {
    name: Ident,
    r#type: Type,
}
impl IterableField for PlainVec {
    fn name(&self) -> &Ident {
        &self.name
    }

    fn r#type(&self) -> &Type {
        &self.r#type
    }

    fn uninitialized(&self) -> TokenStream {
        quote! {
            let mut #(self.name()) = Vec::with_capacity(n_simd_elems);
            #(self.name()).set_len(n_simd_elems);
        }
    }

    fn split_first(&self) -> TokenStream {
        quote! {
            let (
                #(self.name())_head,
                #(self.name())_tail
            ) = self.#(self.name()).split_at_mut(1);
            self.#(self.name()) = #(self.name())_tail;
            let #(self.name()) = &mut (*#(self.name())_head)[0];
        }
    }

    fn split_last(&self) -> TokenStream {
        quote! {
            let (
                #(self.name())_head,
                #(self.name())_tail,
            ) = self.#(self.name()).split_at_mut(#(self.name()).len() - 1);
            self.#(self.name()) = #(self.name())_head;
            let #(self.name()) = &mut (*#(self.name())_tail)[0];
        }
    }

    fn split_at(&self, index: Ident) -> TokenStream {
        quote! {
            let (
                #(self.name()),
                #(self.name())_tail
            ) = self.##(self.name()).split_at_mut(#index);
        }
    }

    fn as_mut_slice(&self) -> TokenStream {
        quote! {
            #(self.name()).as_mut_slice()
        }
    }

    fn len(&self) -> TokenStream {
        quote! {
            #(self.name()).len()
        }
    }

    fn mut_slice_type(&self) -> TokenStream {
        quote! {
            &mut #(self.r#type)
        }
    }

    fn mut_ptr_type(&self) -> TokenStream {
        quote! {
                *mut #(self.r#type)
        }
    }
}

struct ArrayOfVecs {
    name: Ident,
    r#type: Type,
    inner_type: Ident,
    outer_array_size: Expr,
}
impl IterableField for ArrayOfVecs {
    fn name(&self) -> &Ident {
        &self.name
    }

    fn r#type(&self) -> &Type {
        &self.r#type
    }

    fn uninitialized(&self) -> TokenStream {
        quote! {
            let #(self.name()) = [(); #(self.outer_array_size)].map(|_| {
                let mut vec = Vec::with_capacity(n_simd_elems);
                vec.set_len(n_simd_elems);
                vec
            });
        }
    }

    fn split_first(&self) -> TokenStream {
        quote! {
            let #(self.name()) = self.#(self.name()).each_mut().map(|v| {
                let (head, tail) = v.split_at_mut(1);
                *v = tail;
                &mut head[0]
            });
        }
    }

    fn split_last(&self) -> TokenStream {
        quote! {
            let #(self.name()) = self.#(self.name()).each_mut().map(|v| {
                let (head, tail) = v.split_at_mut(v.len() - 1);
                *v = head;
                &mut tail[0]
            });
        }
    }

    fn split_at(&self, index: Ident) -> TokenStream {
        quote! {
            let (
                mut #(self.name()),
                mut #(self.name())_tail
            ) = unsafe { (std::mem::zeroed(), std::mem::zeroed()) };
            self.#(self.name()).into_iter().enumerate().for_each(|(i, v)| {
                let (head, tail) = v.split_at_mut(#index);
                #(self.name())[i] = head;
                #(self.name())_tail[i] = tail;
        });
        }
    }

    fn as_mut_slice(&self) -> TokenStream {
        quote! {
            #(self.name()).each_mut(|v| v.as_mut_slice())
        }
    }

    fn len(&self) -> TokenStream {
        quote! {
            #(self.name())[0].len()
        }
    }

    fn mut_slice_type(&self) -> TokenStream {
        quote! {}
    }

    fn mut_ptr_type(&self) -> TokenStream {
        todo!()
    }
}

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

    let mut iterable_fields: Vec<&dyn IterableField> = vec![];

    let mut first_field_vector = 5;

    for field in fields {
        match field.ty {
            Type::Array(outer_array) => {
                let inner_type = match *outer_array.elem {
                    Type::Path(type_path) => {
                        if let Some(last_segment) = type_path.path.segments.last() {
                            if last_segment.ident != "Vec" {
                                panic!("Expected Vec type");
                            }
                            type_path.path.segments.first().unwrap().ident.clone()
                        } else {
                            panic!("Expected Vec type");
                        }
                    }
                    _ => panic!("Expected Vec type"),
                };
                iterable_fields.push(&ArrayOfVecs {
                    name: field.ident.unwrap(),
                    r#type: field.ty,
                    outer_array_size: outer_array.len,
                    inner_type,
                });
            }
            Type::Path(type_path) => {
                if let Some(last_segment) = type_path.path.segments.last() {
                    if last_segment.ident != "Vec" {
                        panic!("Expected Vec type");
                    }
                }
                iterable_fields.push(&PlainVec {
                    name: field.ident.unwrap(),
                    r#type: field.ty,
                });
            }
            _ => panic!("Expected vector or array of vectors"),
        }
    }

    let mut_chunk_name = format_ident!("{}MutChunk", struct_name);
    let iter_mut_name = format_ident!("{}IterMut", struct_name);
    let row_producer_name = format_ident!("{}RowProducer", struct_name);
    let par_iter_mut_name = format_ident!("Par{}IterMut", struct_name);

    let field_names = iterable_fields.iter().map(|f| f.name()).collect_vec();
    let field_types = iterable_fields.iter().map(|f| f.r#type()).collect_vec();
    let uninitialized_fields = iterable_fields
        .iter()
        .map(|f| f.uninitialized())
        .collect_vec();
    let as_mut_slice = iterable_fields
        .iter()
        .map(|f| f.as_mut_slice())
        .collect_vec();

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
                    #uninitialized_fields,
                )*
                Self {
                    #(#field_names,)*
                }
            }

            pub fn iter_mut(&mut self) -> #iter_mut_name<'_> {
                #iter_mut_name::new(
                    #(#as_mut_slice,)*
                )
            }

            pub fn par_iter_mut(&mut self) -> #par_iter_mut_name<'_> {
                #par_iter_mut_name::new(
                    #(#as_mut_slice,)*
                )
            }
        }

        pub struct #mut_chunk_name<'trace> {
            #(#field_names: #field_types,)*
        }

        pub struct #iter_mut_name<'trace> {

            phantom: std::marker::PhantomData<&'trace ()>,
        }

        impl<'trace> #iter_mut_name<'trace> {
            pub fn new(
                // #(#non_array_field_names: &'trace mut [[PackedM31; #non_array_inner_array_sizes]],)*
                // #(#array_field_names: [&'trace mut [[PackedM31; #inner_array_sizes]]; #outer_array_sizes],)*
            ) -> Self {
                Self {
                    // #(#non_array_field_names: #non_array_field_names as *mut _,)*
                    // #(#array_field_names: #array_field_names.map(|v| v as *mut _),)*
                    // phantom: std::marker::PhantomData,
                }
            }
        }

        impl<'trace> Iterator for #iter_mut_name<'trace> {
            type Item = #mut_chunk_name<'trace>;
            fn next(&mut self) -> Option<Self::Item> {
                if self.#first_field_vector.is_empty() {
                    return None;
                }
                // unsafe {
                //     #(
                        // let (#non_array_field_names_head_prefix, #non_array_field_names_tail_prefix) = self.#non_array_field_names.split_at_mut(1);
                        // self.#non_array_field_names = #non_array_field_names_tail_prefix;
                        // let #non_array_field_names_head_prefix = &mut (*#non_array_field_names_head_prefix)[0];
                    // )*
                    // #(
                        // let #array_field_names = self.#array_field_names.each_mut().map(|ptr| {
                        //     let (head, tail) = ptr.split_at_mut(1);
                        //     *ptr = tail;
                        //     &mut head[0]
                        // });
                    // )*
                    // let item = #mut_chunk_name {
                    //     #(#non_array_field_names)*,
                    //     #(#array_field_names)*
                    // };
                    // Some(item)
                // }
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.#first_field_vector.len();
                (len, Some(len))
            }
        }

        impl ExactSizeIterator for #iter_mut_name<'_> {}

        implDoubleEndedIterator for #iter_mut_name<'_> {
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.#first_field_vector.is_empty() {
                    return None;
                }
                // unsafe {
                //     #(
                //         let (#non_array_field_names_head_prefix, #non_array_field_names_tail_prefix)
                //         = self.#non_array_field_names.split_at_mut(non_array_field_names.len() - 1);
                //         self.#non_array_field_names = #non_array_field_names_head_prefix;
                //         let #non_array_field_names_tail_prefix = &mut (*#non_array_field_names_tail_prefix)[0];
                //     )*
                //     #(
                //         let #array_field_names = self.#array_field_names.each_mut(|ptr| {
                //             let (head, tail) = ptr.split_at_mut(ptr.len() - 1);
                //             *ptr = head;
                //             &mut tail[0]
                //         });
                //     )*
                //     let item = #mut_chunk_name {
                //         #(#non_array_field_names,)*
                //         #(#array_field_names,)*
                //     };
                //     Some(item)
                // }
            }
        }

        pub struct #row_producer_name<'trace> {
            // #(#non_array_field_names: &'trace mut [[PackedM31; #non_array_inner_array_sizes]],)*
            // #(#array_field_names: [&'trace mut [[PackedM31; #inner_array_sizes]]; #outer_array_sizes],)*
        }

        impl<'trace> Producer for  #row_producer_name<'trace> {
            type Item = #mut_chunk_name<'trace>;
            type IntoIter = #iter_mut_name<'trace>;

            #[allow(invalid_value)]
            fn split_at(self, index: usize) -> (Self, Self) {
                // #(
                //     let (#non_array_field_names, #non_array_field_names_tail_prefix) = self.#non_array_field_names.split_at_mut(index);
                // )*
                // #(
                //     let (#array_field_names, #array_field_names_tail_prefix) = self.#array_field_names.map(|v| v.as_mut_slice()).split_at_mut(index);
                // )*
                // (
                //     #row_producer_name { #(#non_array_field_names),* ,#(#array_field_names),* },
                //     #row_producer_name { #(#non_array_field_names_tail_prefix),* ,#(#array_field_names_tail_prefix),* },
                // )

            }

            fn into_iter(self) -> Self::IntoIter {
                // #iter_mut_name::new(#(self.#non_array_field_names),*,#(self.#array_field_names),*)
            }
        }

        pub struct #par_iter_mut_name<'trace> {
            // #(
            //     #non_array_field_names: &'trace mut [[PackedM31; #non_array_inner_array_sizes]],
            // )*,
            // #(
            //     #array_field_names: [&'trace mut [[PackedM31; #inner_array_sizes]]; #outer_array_sizes],
            // )*
        }

        impl<'trace> #par_iter_mut_name<'trace> {
            // pub fn new(
            //     #(#non_array_field_names: &'trace mut [[PackedM31; #non_array_inner_array_sizes]],)*
            //     #(#array_field_names: [&'trace mut [[PackedM31; #inner_array_sizes]]; #outer_array_sizes],)*
            // ) -> Self {
            //     Self { #(#non_array_field_names,)*, #(#array_field_names,)* }
            // }
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
                        // #(#non_array_field_names: self.#non_array_field_names,)*
                        // #(#array_field_names: self.#array_field_names,)*
                    }
                )
            }
        }
    };

    proc_macro::TokenStream::from(expansions)
}

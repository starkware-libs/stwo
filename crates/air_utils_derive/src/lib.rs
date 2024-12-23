use itertools::Itertools;
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

    let mut field_names = vec![];
    let mut array_sizes = vec![];

    for field in fields {
        field_names.push(field.ident.unwrap());

        // Extract array size.
        if let Type::Path(type_path) = field.ty {
            if let Some(last_segment) = type_path.path.segments.last() {
                if let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments {
                    if let Some(syn::GenericArgument::Type(Type::Array(array))) = args.args.first()
                    {
                        array_sizes.push(array.len.clone());
                    }
                }
            }
        }
    }

    let field_names_head = field_names
        .iter()
        .map(|f| format_ident!("head_{}", f))
        .collect_vec();
    let field_names_tail = field_names
        .iter()
        .map(|f| format_ident!("tail_{}", f))
        .collect_vec();
    let first_field = field_names.first().unwrap();

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
                    let mut #field_names = Vec::with_capacity(n_simd_elems);
                    #field_names.set_len(n_simd_elems);
                )*
                Self { #(#field_names),* }
            }

            pub fn iter_mut(&mut self) -> LookupDataIterMut<'_> {
                LookupDataIterMut::new(#(&mut self.#field_names),*)
            }

            pub fn par_iter_mut(&mut self) -> ParLookupDataIterMut<'_> {
                ParLookupDataIterMut { #(#field_names: &mut self.#field_names),* }
            }
        }

        pub struct #mut_chunk_name<'trace> {
            #(pub #field_names: &'trace mut [PackedM31; #array_sizes],)*
        }

        pub struct #iter_mut_name<'trace> {
            #(#field_names: *mut [[PackedM31; #array_sizes]],)*
            phantom: std::marker::PhantomData<&'trace ()>,
        }

        impl<'trace> #iter_mut_name<'trace> {
            pub fn new(#(#field_names: &'trace mut [[PackedM31; #array_sizes]],)*) -> Self {
                Self {
                    #(#field_names: #field_names as *mut _,)*
                    phantom: std::marker::PhantomData,
                }
            }
        }

        impl<'trace> Iterator for #iter_mut_name<'trace> {
            type Item = #mut_chunk_name<'trace>;
            fn next(&mut self) -> Option<Self::Item> {
                if self.#first_field.is_empty()  {
                    return None;
                }
                let item = unsafe {
                    #(
                        let (#field_names_head, #field_names_tail) = (*self.#field_names).split_at_mut(1);
                        self.#field_names = #field_names_tail;
                    )*
                    #mut_chunk_name {
                        #(#field_names: &mut (*#field_names_head)[0],)*
                    }
                };
                Some(item)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.#first_field.len();
                (len, Some(len))
            }
        }

        impl ExactSizeIterator for #iter_mut_name<'_> {}

        impl<'trace> DoubleEndedIterator for #iter_mut_name<'trace> {
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.#first_field.is_empty() {
                    return None;
                }
                let item = unsafe {
                    #(
                        let (#field_names_head, #field_names_tail) = (*self.#field_names)
                            .split_at_mut(self.#field_names.len() - 1);
                        self.#field_names = #field_names_head;
                    )*
                    #mut_chunk_name {
                        #(#field_names: &mut (*#field_names_tail)[0],)*
                    }
                };
                Some(item)
            }
        }

        pub struct #row_producer_name<'trace> {
            #(#field_names: &'trace mut [[PackedM31; #array_sizes]],)*
        }

        impl<'trace> Producer for  #row_producer_name<'trace> {
            type Item = #mut_chunk_name<'trace>;
            type IntoIter = #iter_mut_name<'trace>;

            fn split_at(self, index: usize) -> (Self, Self) {
                #(
                    let (#field_names, #field_names_tail) = self.#field_names.split_at_mut(index);
                )*
                (
                    Self { #(#field_names,)* },
                    Self { #(#field_names: #field_names_tail,)* }
                )
            }

            fn into_iter(self) -> Self::IntoIter {
                #iter_mut_name::new(#(self.#field_names),*)
            }
        }

        pub struct #par_iter_mut_name<'trace> {
            #(#field_names: &'trace mut [[PackedM31; #array_sizes]],)*
        }

        impl<'trace> #par_iter_mut_name<'trace> {
            pub fn new(#(#field_names: &'trace mut [[PackedM31; #array_sizes]],)*) -> Self {
                Self { #(#field_names,)* }
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
                self.#first_field.len()
            }

            fn drive<D: Consumer<Self::Item>>(self, consumer: D) -> D::Result {
                bridge(self, consumer)
            }

            fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
                callback.callback(#row_producer_name { #(#field_names: self.#field_names),* })
            }
        }
    };

    proc_macro::TokenStream::from(expansions)
}

#![allow(dead_code)]
#![allow(unused_variables)]
mod iterable_field;
use iterable_field::IterableField;
use itertools::Itertools;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Data, DeriveInput, Fields, Lifetime};

#[proc_macro_derive(Iterable)]
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

    let iterable_fields = fields.iter().map(IterableField::from_field).collect_vec();

    let mut_chunk_name = format_ident!("{}MutChunk", struct_name);
    let iter_mut_name = format_ident!("{}IterMut", struct_name);
    let row_producer_name = format_ident!("{}RowProducer", struct_name);
    let par_iter_mut_name = format_ident!("Par{}IterMut", struct_name);

    let (
        field_names,
        field_types,
        uninitialized_fields,
        as_mut_slice,
        as_mut_ptr,
        field_ptr_types,
        mut_slice_types,
        mut_chunk_types,
        split_first,
        split_last,
        split_at,
    ): (
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
    ) = iterable_fields
        .iter()
        .map(|f| {
            (
                f.name(),
                f.r#type(),
                f.uninitialized(),
                f.as_mut_slice(),
                f.as_mut_ptr(),
                f.mut_ptr_type(),
                f.mut_slice_type(&Lifetime::new("'trace", Span::call_site())),
                f.mut_chunk_type(&Lifetime::new("'trace", Span::call_site())),
                f.split_first(),
                f.split_last(&format_ident!("len")),
                f.split_at(format_ident!("index")),
            )
        })
        .multiunzip();
    let get_length = iterable_fields.first().unwrap().get_len();

    let field_names_tail = field_names
        .iter()
        .map(|f| format_ident!("{}_tail", f))
        .collect_vec();

    let expansions = quote! {
        impl #struct_name {
            /// # Safety
            /// The caller must ensure that the trace is populated before being used.
            #[allow(clippy::uninit_vec)]
            pub unsafe fn uninitialized(log_size: u32) -> Self {
                let len = 1 << log_size;
                #(#uninitialized_fields)*
                Self {
                    #(#field_names,)*
                }
            }

            pub fn iter_mut(&mut self) -> #iter_mut_name<'_> {
                #iter_mut_name::new(
                    #(self.#as_mut_slice,)*
                )
            }

            pub fn par_iter_mut(&mut self) -> #par_iter_mut_name<'_> {
                #par_iter_mut_name::new(
                    #(self.#as_mut_slice,)*
                )
            }
        }

        pub struct #mut_chunk_name<'trace> {
            #(#field_names: #mut_chunk_types,)*
        }

        pub struct #iter_mut_name<'trace> {
            #(#field_names: #field_ptr_types,)*
            phantom: std::marker::PhantomData<&'trace ()>,
        }

        impl<'trace> #iter_mut_name<'trace> {
            pub fn new(
                #(#field_names: #mut_slice_types,)*
            ) -> Self {
                Self {
                    #(#field_names: #as_mut_ptr,)*
                    phantom: std::marker::PhantomData,
                }
            }
        }

        impl<'trace> Iterator for #iter_mut_name<'trace> {
            type Item = #mut_chunk_name<'trace>;
            fn next(&mut self) -> Option<Self::Item> {
                if self.#get_length == 0 {
                    return None;
                }
                let item = unsafe {
                    #(#split_first)*
                    #mut_chunk_name {
                        #(#field_names,)*
                    }
                };
                Some(item)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.#get_length;
                (len, Some(len))
            }
        }

        impl ExactSizeIterator for #iter_mut_name<'_> {}
        impl DoubleEndedIterator for #iter_mut_name<'_> {
            fn next_back(&mut self) -> Option<Self::Item> {
                let len = self.#get_length;
                if len == 0 {
                    return None;
                }
                let item = unsafe {
                    #(#split_last)*
                    #mut_chunk_name {
                        #(#field_names,)*
                    }
                };
                Some(item)
            }
        }

        pub struct #row_producer_name<'trace> {
            #(#field_names: #mut_slice_types,)*
        }

        impl<'trace> Producer for  #row_producer_name<'trace> {
            type Item = #mut_chunk_name<'trace>;
            type IntoIter = #iter_mut_name<'trace>;

            #[allow(invalid_value)]
            fn split_at(self, index: usize) -> (Self, Self) {
                #(#split_at)*
                (
                    #row_producer_name {
                        #(#field_names,)*
                    },
                    #row_producer_name {
                        #(#field_names: #field_names_tail,)*
                    }
                )

            }

            fn into_iter(self) -> Self::IntoIter {
                #iter_mut_name::new(#(self.#field_names),*)
            }
        }

        pub struct #par_iter_mut_name<'trace> {
            #(#field_names: #mut_slice_types,)*
        }

        impl<'trace> #par_iter_mut_name<'trace> {
            pub fn new(
                #(#field_names: #mut_slice_types,)*
            ) -> Self {
                Self {
                    #(#field_names,)*
                }
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
                self.#get_length
            }

            fn drive<D: Consumer<Self::Item>>(self, consumer: D) -> D::Result {
                bridge(self, consumer)
            }

            fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
                callback.callback(
                    #row_producer_name {
                        #(#field_names : self.#field_names,)*
                    }
                )
            }
        }
    };

    proc_macro::TokenStream::from(expansions)
}

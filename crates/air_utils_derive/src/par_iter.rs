use itertools::Itertools;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{Ident, Lifetime};

use crate::iterable_field::IterableField;

#[allow(clippy::too_many_arguments)]
pub fn expand_par_iter_mut_structs(
    struct_name: &Ident,
    iterable_fields: &[IterableField],
) -> TokenStream {
    let lifetime = Lifetime::new("'a", Span::call_site());
    let (
        field_names,
        mut_chunk_types,
        mut_slice_types,
        mut_ptr_types,
        as_mut_slice,
        as_mut_ptr,
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
    ) = iterable_fields
        .iter()
        .map(|f| {
            (
                f.name(),
                f.mut_chunk_type(&lifetime),
                f.mut_slice_type(&lifetime),
                f.mut_ptr_type(),
                f.as_mut_slice(),
                f.as_mut_ptr(),
                f.split_first(),
                f.split_last(&format_ident!("len")),
                f.split_at(format_ident!("index")),
            )
        })
        .multiunzip();
    let get_length = iterable_fields.first().unwrap().get_len();

    let iter_mut_name = format_ident!("{}IterMut", struct_name);
    let mut_chunk_name = format_ident!("{}MutChunk", struct_name);
    let row_producer_name = format_ident!("{}RowProducer", struct_name);
    let par_iter_mut_name = format_ident!("{}ParIterMut", struct_name);

    let field_names_tail = field_names
        .iter()
        .map(|f| format_ident!("{}_tail", f))
        .collect_vec();

    quote! {
        impl #struct_name {
            pub fn par_iter_mut(&mut self) -> #par_iter_mut_name<'_> {
                #par_iter_mut_name::new(
                    #(self.#as_mut_slice,)*
                )
            }
        }

        pub struct #row_producer_name<#lifetime> {
            #(#field_names: #mut_slice_types,)*
        }

        impl<#lifetime> Producer for  #row_producer_name<#lifetime> {
            type Item = #mut_chunk_name<#lifetime>;
            type IntoIter = #iter_mut_name<#lifetime>;

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

        pub struct #par_iter_mut_name<#lifetime> {
            #(#field_names: #mut_slice_types,)*
        }

        impl<#lifetime> #par_iter_mut_name<#lifetime> {
            pub fn new(
                #(#field_names: #mut_slice_types,)*
            ) -> Self {
                Self {
                    #(#field_names,)*
                }
            }
        }

        impl<#lifetime> ParallelIterator for #par_iter_mut_name<#lifetime> {
            type Item = #mut_chunk_name<#lifetime>;

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
    }
}

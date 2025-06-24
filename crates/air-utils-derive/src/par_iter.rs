use itertools::Itertools;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{Ident, Lifetime};

use crate::iterable_field::IterableField;

pub fn expand_par_iter_mut_structs(
    struct_name: &Ident,
    iterable_fields: &[IterableField],
) -> TokenStream {
    let lifetime = Lifetime::new("'a", Span::call_site());
    let split_index = format_ident!("index");

    let struct_code = generate_struct_impl(struct_name, iterable_fields);
    let producer_code =
        generate_row_producer(struct_name, iterable_fields, &lifetime, &split_index);
    let oar_iter_struct = generate_par_iter_struct(struct_name, iterable_fields, &lifetime);
    let impl_par_iter = generate_parallel_iterator_impls(struct_name, iterable_fields, &lifetime);

    quote! {
        #struct_code
        #producer_code
        #oar_iter_struct
        #impl_par_iter
    }
}

fn generate_struct_impl(struct_name: &Ident, iterable_fields: &[IterableField]) -> TokenStream {
    let par_iter_mut_name = format_ident!("{}ParIterMut", struct_name);
    let as_mut_slice = iterable_fields.iter().map(|f| f.as_mut_slice());
    quote! {
        impl #struct_name {
            pub fn par_iter_mut(&mut self) -> #par_iter_mut_name<'_> {
                #par_iter_mut_name::new(
                    #(self.#as_mut_slice,)*
                )
            }
        }
    }
}

fn generate_row_producer(
    struct_name: &Ident,
    iterable_fields: &[IterableField],
    lifetime: &Lifetime,
    split_index: &Ident,
) -> TokenStream {
    let row_producer_name = format_ident!("{}RowProducer", struct_name);
    let mut_chunk_name = format_ident!("{}MutChunk", struct_name);
    let iter_mut_name = format_ident!("{}IterMut", struct_name);
    let (field_names, mut_slice_types, split_at): (Vec<_>, Vec<_>, Vec<_>) = iterable_fields
        .iter()
        .map(|f| {
            (
                f.name(),
                f.mut_slice_type(lifetime),
                f.split_at(split_index),
            )
        })
        .multiunzip();
    let field_names_head = field_names.iter().map(|f| format_ident!("{}_head", f));
    let field_names_tail = field_names.iter().map(|f| format_ident!("{}_tail", f));
    quote! {
        pub struct #row_producer_name<#lifetime> {
            #(#field_names: #mut_slice_types,)*
        }
        impl<#lifetime> rayon::iter::plumbing::Producer for #row_producer_name<#lifetime> {
            type Item = #mut_chunk_name<#lifetime>;
            type IntoIter = #iter_mut_name<#lifetime>;

            #[allow(invalid_value)]
            fn split_at(self, index: usize) -> (Self, Self) {
                #(#split_at)*
                (
                    #row_producer_name {
                        #(#field_names: #field_names_head,)*
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
    }
}

fn generate_par_iter_struct(
    struct_name: &Ident,
    iterable_fields: &[IterableField],
    lifetime: &Lifetime,
) -> TokenStream {
    let par_iter_mut_name = format_ident!("{struct_name}ParIterMut");
    let (field_names, mut_slice_types): (Vec<_>, Vec<_>) = iterable_fields
        .iter()
        .map(|f| (f.name(), f.mut_slice_type(lifetime)))
        .unzip();
    quote! {
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
    }
}

fn generate_parallel_iterator_impls(
    struct_name: &Ident,
    iterable_fields: &[IterableField],
    lifetime: &Lifetime,
) -> TokenStream {
    let par_iter_mut_name = format_ident!("{}ParIterMut", struct_name);
    let mut_chunk_name = format_ident!("{}MutChunk", struct_name);
    let row_producer_name = format_ident!("{}RowProducer", struct_name);
    let field_names = iterable_fields.iter().map(|f| f.name());
    let get_length = iterable_fields.first().unwrap().get_len();
    quote! {
        impl<#lifetime> rayon::prelude::ParallelIterator for #par_iter_mut_name<#lifetime> {
            type Item = #mut_chunk_name<#lifetime>;

            fn drive_unindexed<D>(self, consumer: D) -> D::Result
            where
                D: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
            {
                rayon::iter::plumbing::bridge(self, consumer)
            }

            fn opt_len(&self) -> Option<usize> {
                Some(self.len())
            }
        }

        impl rayon::iter::IndexedParallelIterator for #par_iter_mut_name<'_> {
            fn len(&self) -> usize {
                self.#get_length
            }

            fn drive<D: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: D) -> D::Result {
                rayon::iter::plumbing::bridge(self, consumer)
            }

            fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
                callback.callback(
                    #row_producer_name {
                        #(#field_names : self.#field_names,)*
                    }
                )
            }
        }
    }
}

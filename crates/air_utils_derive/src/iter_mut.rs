use itertools::Itertools;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{Ident, Lifetime};

use crate::iterable_field::IterableField;

pub fn expand_iter_mut_structs(
    struct_name: &Ident,
    iterable_fields: &[IterableField],
) -> TokenStream {
    let impl_struct_name = expand_impl_struct_name(struct_name, iterable_fields);
    let mut_chunk_struct = expand_mut_chunk_struct(struct_name, iterable_fields);
    let iter_mut_struct = expand_iter_mut_struct(struct_name, iterable_fields);
    let iterator_impl = expand_iterator_impl(struct_name, iterable_fields);
    let exact_size_iterator = expand_exact_size_iterator(struct_name);
    let double_ended_iterator = expand_double_ended_iterator(struct_name, iterable_fields);

    quote! {
        #impl_struct_name
        #mut_chunk_struct
        #iter_mut_struct
        #iterator_impl
        #exact_size_iterator
        #double_ended_iterator
    }
}

fn expand_impl_struct_name(struct_name: &Ident, iterable_fields: &[IterableField]) -> TokenStream {
    let iter_mut_name = format_ident!("{}IterMut", struct_name);
    let as_mut_slice = iterable_fields
        .iter()
        .map(|f| f.as_mut_slice())
        .collect_vec();
    quote! {
        impl #struct_name {
            pub fn iter_mut(&mut self) -> #iter_mut_name<'_> {
                #iter_mut_name::new(
                    #(self.#as_mut_slice,)*
                )
            }
        }
    }
}

fn expand_mut_chunk_struct(struct_name: &Ident, iterable_fields: &[IterableField]) -> TokenStream {
    let lifetime = Lifetime::new("'a", Span::call_site());
    let mut_chunk_name = format_ident!("{}MutChunk", struct_name);
    let (field_names, mut_chunk_types): (Vec<_>, Vec<_>) = iterable_fields
        .iter()
        .map(|f| (f.name(), f.mut_chunk_type(&lifetime)))
        .unzip();

    quote! {
        pub struct #mut_chunk_name<#lifetime> {
            #(#field_names: #mut_chunk_types,)*
        }
    }
}

fn expand_iter_mut_struct(struct_name: &Ident, iterable_fields: &[IterableField]) -> TokenStream {
    let lifetime = Lifetime::new("'a", Span::call_site());
    let iter_mut_name = format_ident!("{}IterMut", struct_name);
    let (field_names, mut_slice_types, mut_ptr_types, as_mut_ptr): (
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
    ) = iterable_fields
        .iter()
        .map(|f| {
            (
                f.name(),
                f.mut_slice_type(&lifetime),
                f.mut_slice_ptr_type(),
                f.as_mut_ptr(),
            )
        })
        .multiunzip();

    quote! {
        pub struct #iter_mut_name<#lifetime> {
            #(#field_names: #mut_ptr_types,)*
            phantom: std::marker::PhantomData<&#lifetime ()>,
        }
        impl<#lifetime> #iter_mut_name<#lifetime> {
            pub fn new(
                #(#field_names: #mut_slice_types,)*
            ) -> Self {
                Self {
                    #(#field_names: #as_mut_ptr,)*
                    phantom: std::marker::PhantomData,
                }
            }
        }
    }
}

fn expand_iterator_impl(struct_name: &Ident, iterable_fields: &[IterableField]) -> TokenStream {
    let lifetime = Lifetime::new("'a", Span::call_site());
    let iter_mut_name = format_ident!("{}IterMut", struct_name);
    let mut_chunk_name = format_ident!("{}MutChunk", struct_name);
    let (field_names, split_first): (Vec<_>, Vec<_>) = iterable_fields
        .iter()
        .map(|f| (f.name(), f.split_first()))
        .unzip();
    let get_length = iterable_fields.first().unwrap().get_len();

    quote! {
        impl<#lifetime> Iterator for #iter_mut_name<#lifetime> {
            type Item = #mut_chunk_name<#lifetime>;
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
    }
}

fn expand_exact_size_iterator(struct_name: &Ident) -> TokenStream {
    let iter_mut_name = format_ident!("{}IterMut", struct_name);
    quote! {
        impl ExactSizeIterator for #iter_mut_name<'_> {}
    }
}

fn expand_double_ended_iterator(
    struct_name: &Ident,
    iterable_fields: &[IterableField],
) -> TokenStream {
    let iter_mut_name = format_ident!("{}IterMut", struct_name);
    let mut_chunk_name = format_ident!("{}MutChunk", struct_name);
    let (field_names, split_last): (Vec<_>, Vec<_>) = iterable_fields
        .iter()
        .map(|f| (f.name(), f.split_last(&format_ident!("len"))))
        .unzip();
    let get_length = iterable_fields.first().unwrap().get_len();
    quote! {
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
    }
}

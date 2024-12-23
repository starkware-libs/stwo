use itertools::Itertools;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{Ident, Lifetime};

use crate::iterable_field::IterableField;

pub fn expand_iter_mut_structs(
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
    ): (
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
                f.mut_slice_ptr_type(),
                f.as_mut_slice(),
                f.as_mut_ptr(),
                f.split_first(),
                f.split_last(&format_ident!("len")),
            )
        })
        .multiunzip();
    let get_length = iterable_fields.first().unwrap().get_len();

    let iter_mut_name = format_ident!("{}IterMut", struct_name);
    let mut_chunk_name = format_ident!("{}MutChunk", struct_name);

    quote! {
        impl #struct_name {
            pub fn iter_mut(&mut self) -> #iter_mut_name<'_> {
                #iter_mut_name::new(
                    #(self.#as_mut_slice,)*
                )
            }
        }

        pub struct #mut_chunk_name<#lifetime> {
            #(#field_names: #mut_chunk_types,)*
        }

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
    }
}

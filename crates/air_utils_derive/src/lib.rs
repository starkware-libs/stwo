#![allow(dead_code)]
#![allow(unused_variables)]
mod allocation;
mod iter_mut;
mod iterable_field;
mod par_iter;
use iterable_field::IterableField;
use itertools::Itertools;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

#[proc_macro_derive(Iterable)]
pub fn derive_stwo_iterable(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = &input.ident;
    let input = match input.data {
        Data::Struct(data_struct) => data_struct,
        _ => {
            return syn::Error::new_spanned(struct_name, "Expected struct")
                .to_compile_error()
                .into()
        }
    };

    let fields = match input.fields {
        Fields::Named(fields) => fields.named,
        _ => {
            return syn::Error::new_spanned(struct_name, "Expected named fields")
                .to_compile_error()
                .into()
        }
    };

    let iterable_fields = fields.iter().map(IterableField::from_field).collect_vec();

    let expand_uninitialized = allocation::expand_uninitialized_impl(struct_name, &iterable_fields);
    let expand_iter_mut = iter_mut::expand_iter_mut_structs(struct_name, &iterable_fields);
    let expand_par_iter_mut = par_iter::expand_par_iter_mut_structs(struct_name, &iterable_fields);

    // TODO(Ohad): consider separating into different macros.

    proc_macro::TokenStream::from(quote! {
        #expand_uninitialized
        #expand_iter_mut
        #expand_par_iter_mut
    })
}

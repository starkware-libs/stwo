mod allocation;
mod iter_mut;
mod iterable_field;
mod par_iter;
use iterable_field::to_iterable_fields;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(Uninitialized)]
pub fn derive_uninitialized(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = input.ident.clone();

    let iterable_fields = match to_iterable_fields(input) {
        Ok(iterable_fields) => iterable_fields,
        Err(err) => return err.into_compile_error().into(),
    };

    allocation::expand_uninitialized_impl(&struct_name, &iterable_fields).into()
}

#[proc_macro_derive(IterMut)]
pub fn derive_mut_iter(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = input.ident.clone();

    let iterable_fields = match to_iterable_fields(input) {
        Ok(iterable_fields) => iterable_fields,
        Err(err) => return err.into_compile_error().into(),
    };

    iter_mut::expand_iter_mut_structs(&struct_name, &iterable_fields).into()
}

#[proc_macro_derive(ParIterMut)]
pub fn derive_par_mut_iter(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = input.ident.clone();

    let iterable_fields = match to_iterable_fields(input) {
        Ok(iterable_fields) => iterable_fields,
        Err(err) => return err.into_compile_error().into(),
    };

    par_iter::expand_par_iter_mut_structs(&struct_name, &iterable_fields).into()
}

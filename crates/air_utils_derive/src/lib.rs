mod allocation;
mod iter_mut;
mod iterable_field;
use iterable_field::IterableField;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

#[proc_macro_derive(Uninitialized)]
pub fn derive_uninitialized(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = input.ident.clone();

    let iterable_fields = match parse_derive_input_to_iterable_fields(input) {
        Ok(iterable_fields) => iterable_fields,
        Err(err) => return err.into_compile_error().into(),
    };

    allocation::expand_uninitialized_impl(&struct_name, &iterable_fields).into()
}

#[proc_macro_derive(MutIter)]
pub fn derive_mut_iter(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = input.ident.clone();

    let iterable_fields = match parse_derive_input_to_iterable_fields(input) {
        Ok(iterable_fields) => iterable_fields,
        Err(err) => return err.into_compile_error().into(),
    };

    iter_mut::expand_iter_mut_structs(&struct_name, &iterable_fields).into()
}

fn parse_derive_input_to_iterable_fields(
    input: DeriveInput,
) -> Result<Vec<IterableField>, syn::Error> {
    let struct_name = &input.ident;
    let input = match input.data {
        Data::Struct(data_struct) => Ok(data_struct),
        _ => Err(syn::Error::new_spanned(struct_name, "Expected a struct")),
    }?;

    match input.fields {
        Fields::Named(fields) => Ok(fields
            .named
            .iter()
            .map(IterableField::from_field)
            .collect::<Result<_, _>>()?),
        _ => Err(syn::Error::new_spanned(
            input.fields,
            "Expected named fields",
        )),
    }
}

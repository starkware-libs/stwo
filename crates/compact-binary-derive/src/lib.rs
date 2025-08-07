use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

/// Proc macro to automatically derive `CompactBinary` trait for structs.
#[proc_macro_derive(CompactBinary)]
pub fn derive_compact_binary(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree.
    let input = parse_macro_input!(input as DeriveInput);

    let struct_name = input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Extract the fields of the struct.
    let fields = match input.data {
        Data::Struct(ref data_struct) => match &data_struct.fields {
            Fields::Named(ref fields_named) => &fields_named.named,
            Fields::Unnamed(_) | Fields::Unit => {
                return syn::Error::new_spanned(
                    struct_name,
                    "CompactBinary can only be derived for structs with named fields.",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(
                struct_name,
                "CompactBinary can only be derived for structs.",
            )
            .to_compile_error()
            .into();
        }
    };

    // Generate code to serialize each field in the order they appear.
    let compact_serialize_body = fields.iter().enumerate().map(|(i, f)| {
        let field_name = &f.ident;
        quote! {
            usize::compact_serialize(&#i, output)?;
            CompactBinary::compact_serialize(&self.#field_name, output)?;
        }
    });

    // Generate code to deserialize each field in the order they appear.
    let compact_deserialize_let_bindings = fields.iter().enumerate().map(|(i, f)| {
        let field_name = &f.ident;
        quote! {
            let input = strip_expected_tag(input, #i)?;
            let (input, #field_name) = stwo::core::compact_binary::CompactBinary::compact_deserialize(input)?;
        }
    });
    let compact_deserialize_struct_fields = fields.iter().map(|f| {
        let field_name = &f.ident;
        quote! { #field_name }
    });

    // Implement `CompactBinary` for the type.
    let expanded = quote! {
        impl #impl_generics stwo::core::compact_binary::CompactBinary for #struct_name #ty_generics #where_clause {
            fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), CompactSerializeError> {
                u32::compact_serialize(&0, output)?;
                #(#compact_serialize_body)*
            }

            fn compact_deserialize<'a>(mut input: &'a [u8]) -> (&'a [u8], Self) {
                let input = strip_expected_version(input, 0)?;
                #(#compact_deserialize_let_bindings)*
                (input, Self { #(#compact_deserialize_struct_fields),* })
            }
        }
    };

    TokenStream::from(expanded)
}

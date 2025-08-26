use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, parse_quote, Data, DeriveInput, Fields, Type};

/// Proc macro to automatically derive `CompactBinary` trait for structs.
#[proc_macro_derive(CompactBinary, attributes(zipped))]
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

    // Check if MerkleHasher is present in the where clause or generics
    let h_is_merklehasher = input
        .generics
        .where_clause
        .as_ref()
        .map(|wc| {
            wc.predicates.iter().any(|pred| {
                pred.to_token_stream()
                    .to_string()
                    .contains("H: MerkleHasher")
            })
        })
        .unwrap_or(false)
        || input.generics.params.iter().any(|param| {
            if let syn::GenericParam::Type(ty) = param {
                ty.bounds
                    .iter()
                    .any(|b| b.to_token_stream().to_string().contains("MerkleHasher"))
            } else {
                false
            }
        });

    // Check if any field requires H bounds
    let needs_h_bounds = fields.iter().any(|f| {
        if let Type::Path(type_path) = &f.ty {
            if let Some(seg) = type_path.path.segments.last() {
                if let syn::PathArguments::AngleBracketed(ref args) = seg.arguments {
                    return args.args.iter().any(|arg| {
                        if let syn::GenericArgument::Type(Type::Path(type_path)) = arg {
                            type_path
                                .path
                                .segments
                                .last()
                                .is_some_and(|s| s.ident == "H")
                        } else {
                            false
                        }
                    });
                }
            }
        }
        false
    });

    let mut where_clause = where_clause.cloned();
    // If MerkleHasher is present and H bounds are needed, add the necessary bounds.
    if h_is_merklehasher && needs_h_bounds {
        let pred: syn::WherePredicate =
            parse_quote! { H::Hash: stwo::core::compact_binary::CompactBinary };
        if let Some(ref mut wc) = where_clause {
            wc.predicates.push(pred);
        } else {
            where_clause = Some::<syn::WhereClause>(
                parse_quote! { where H::Hash: stwo::core::compact_binary::CompactBinary },
            );
        }
    }

    // Generate code to serialize each field in the order they appear.
    let compact_serialize_body = fields.iter().enumerate().map(|(i, f)| {
        let field_name = &f.ident;
        let field_type = &f.ty;
        let is_zipped = f.attrs.iter().any(|attr| attr.path().is_ident("zipped"));
        match is_zipped {
            true => {
                quote! {
                    usize::compact_serialize(&#i, output)?;
                    let #field_name = stwo::core::compact_binary::ZippedCompactBinary(&self.#field_name);
                    stwo::core::compact_binary::ZippedCompactBinary::<&#field_type>::compact_serialize(&#field_name, output)?;
                }
            }
            false => {
                quote! {
                    usize::compact_serialize(&#i, output)?;
                    stwo::core::compact_binary::CompactBinary::compact_serialize(&self.#field_name, output)?;
                }
            }
        }
    });

    // Generate code to deserialize each field in the order they appear.
    let compact_deserialize_let_bindings = fields.iter().enumerate().map(|(i, f)| {
        let field_name = &f.ident;
        let field_type = &f.ty;
        let is_zipped = f.attrs.iter().any(|attr| attr.path().is_ident("zipped"));
        match is_zipped {
            true => {
                quote! {
                    let input = stwo::core::compact_binary::strip_expected_tag(input, #i)?;
                    let (input, #field_name) = stwo::core::compact_binary::ZippedCompactBinary::<&#field_type>::compact_deserialize(input)?;
                }
            }
            false => {
                quote! {
                    let input = stwo::core::compact_binary::strip_expected_tag(input, #i)?;
                    let (input, #field_name) = stwo::core::compact_binary::CompactBinary::compact_deserialize(input)?;
                }
            }
        }
    });
    let compact_deserialize_struct_fields = fields.iter().map(|f| {
        let field_name = &f.ident;
        quote! { #field_name }
    });

    // Implement `CompactBinary` for the type.
    let expanded = quote! {
        impl #impl_generics stwo::core::compact_binary::CompactBinary for #struct_name #ty_generics #where_clause {
            fn compact_serialize(&self, output: &mut Vec<u8>) -> Result<(), stwo::core::compact_binary::CompactSerializeError> {
                u32::compact_serialize(&0, output)?;
                #(#compact_serialize_body)*
                Ok(())
            }

            fn compact_deserialize<'a>(mut input: &'a [u8]) -> Result<(&'a [u8], Self), stwo::core::compact_binary::CompactDeserializeError> {
                let input = stwo::core::compact_binary::strip_expected_version(input, 0)?;
                #(#compact_deserialize_let_bindings)*
                Ok((input, Self { #(#compact_deserialize_struct_fields),* }))
            }
        }
    };

    TokenStream::from(expanded)
}

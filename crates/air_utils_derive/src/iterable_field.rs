use proc_macro2::TokenStream;
use quote::quote;
use syn::{Expr, Field, Ident, Type};

/// Each variant represents a field that can be iterated over.
/// Used to derive implementations of `Uninitialized`, `MutIter`, and `ParMutIter`.
/// Currently supports `Vec<T>` and `[Vec<T>; N]` fields only.
pub(super) enum IterableField {
    /// A single Vec<T> field, e.g. `Vec<u32>`, `Vec<[u32; K]>`.
    PlainVec(PlainVec),
    /// An array of Vec<T> fields, e.g. `[Vec<u32>; N]`, `[Vec<[u32; K]>; N]`.
    ArrayOfVecs(ArrayOfVecs),
}

pub(super) struct PlainVec {
    name: Ident,
    _ty: Type,
}
pub(super) struct ArrayOfVecs {
    name: Ident,
    _inner_type: Type,
    outer_array_size: Expr,
}

impl IterableField {
    pub fn from_field(field: &Field) -> Result<Self, syn::Error> {
        // Check if the field is a vector or array of vectors.
        match field.ty {
            Type::Array(ref outer_array) => {
                // Assert that the inner type is a Vec<T>.
                let inner_type = match outer_array.elem.as_ref() {
                    Type::Path(ref type_path) => parse_inner_type(type_path)?,
                    _ => Err(syn::Error::new_spanned(
                        outer_array.elem.clone(),
                        "Expected Vec<T> type",
                    ))?,
                };
                Ok(Self::ArrayOfVecs(ArrayOfVecs {
                    name: field.ident.clone().unwrap(),
                    outer_array_size: outer_array.len.clone(),
                    _inner_type: inner_type.clone(),
                }))
            }
            // Assert that the type is a Vec<T>.
            Type::Path(ref type_path) => {
                let _ty = parse_inner_type(type_path)?;
                Ok(Self::PlainVec(PlainVec {
                    name: field.ident.clone().unwrap(),
                    _ty,
                }))
            }
            _ => Err(syn::Error::new_spanned(
                field,
                "Expected vector or array of vectors",
            )),
        }
    }

    pub fn name(&self) -> &Ident {
        match self {
            IterableField::PlainVec(plain_vec) => &plain_vec.name,
            IterableField::ArrayOfVecs(array_of_vecs) => &array_of_vecs.name,
        }
    }

    /// Generate the uninitialized allocation for the field.
    /// E.g. `Vec::with_capacity(len); vec.set_len(len);` for a `Vec<u32>` field.
    /// E.g. `[(); N].map(|_| { Vec::with_capacity(len); vec.set_len(len); })` for `[Vec<u32>; N]`.
    pub fn uninitialized_field_allocation(&self) -> TokenStream {
        match self {
            IterableField::PlainVec(plain_vec) => {
                let name = &plain_vec.name;
                quote! {
                    let mut #name= Vec::with_capacity(len);
                    #name.set_len(len);
                }
            }
            IterableField::ArrayOfVecs(array_of_vecs) => {
                let name = &array_of_vecs.name;
                let outer_array_size = &array_of_vecs.outer_array_size;
                quote! {
                    let #name = [(); #outer_array_size].map(|_| {
                        let mut vec = Vec::with_capacity(len);
                        vec.set_len(len);
                        vec
                    });
                }
            }
        }
    }
}

// Extract the inner vector type from a Vec<T> or [Vec<T>; N] type:
fn parse_inner_type(type_path: &syn::TypePath) -> Result<Type, syn::Error> {
    match type_path.path.segments.last() {
        Some(last_segment) => {
            if last_segment.ident != "Vec" {
                return Err(syn::Error::new_spanned(
                    last_segment.ident.clone(),
                    "Expected Vec<T> type",
                ));
            }
            match &last_segment.arguments {
                syn::PathArguments::AngleBracketed(args) => match args.args.first() {
                    Some(syn::GenericArgument::Type(inner_type)) => Ok(inner_type.clone()),
                    _ => Err(syn::Error::new_spanned(
                        args.args.first().unwrap(),
                        "Expected exactly one generic argument: Vec<T>",
                    )),
                },
                _ => Err(syn::Error::new_spanned(
                    last_segment.arguments.clone(),
                    "Expected angle-bracketed arguments: Vec<..>",
                )),
            }
        }
        None => Err(syn::Error::new_spanned(
            type_path.path.clone(),
            "Expected last segment",
        )),
    }
}

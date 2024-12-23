use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Expr, Field, Ident, Lifetime, Type};

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
    ty: Type,
}
pub(super) struct ArrayOfVecs {
    name: Ident,
    inner_type: Type,
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
                    inner_type: inner_type.clone(),
                }))
            }
            // Assert that the type is a Vec<T>.
            Type::Path(ref type_path) => {
                let ty = parse_inner_type(type_path)?;
                Ok(Self::PlainVec(PlainVec {
                    name: field.ident.clone().unwrap(),
                    ty,
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

    /// Generate the type of a mutable slice of the field.
    /// E.g. `&'a mut [u32]` for a `Vec<u32>` field.
    /// E.g. [`&'a mut u32; N]` for a `[Vec<u32>; N]` field.
    pub fn mut_slice_type(&self, lifetime: &Lifetime) -> TokenStream {
        match self {
            IterableField::PlainVec(plain_vec) => {
                let ty = &plain_vec.ty;
                quote! {
                    &#lifetime mut [#ty]
                }
            }
            IterableField::ArrayOfVecs(array_of_vecs) => {
                let inner_type = &array_of_vecs.inner_type;
                let outer_array_size = &array_of_vecs.outer_array_size;
                quote! {
                    [&#lifetime mut [#inner_type]; #outer_array_size]
                }
            }
        }
    }

    /// Generate the type of a mutable chunk of the field.
    /// E.g. `&'a mut u32` for a `Vec<u32>` field.
    /// E.g. [`&'a mut u32; N]` for a `[Vec<u32>; N]` field.
    pub fn mut_chunk_type(&self, lifetime: &Lifetime) -> TokenStream {
        match self {
            IterableField::PlainVec(plain_vec) => {
                let ty = &plain_vec.ty;
                quote! {
                    &#lifetime mut #ty
                }
            }
            IterableField::ArrayOfVecs(array_of_vecs) => {
                let inner_type = &array_of_vecs.inner_type;
                let array_size = &array_of_vecs.outer_array_size;
                quote! {
                    [&#lifetime mut #inner_type; #array_size]
                }
            }
        }
    }

    /// Generate the type of a mutable slice pointer to the field.
    /// E.g. `*mut [u32]` for a `Vec<u32>` field.
    /// E.g. [`*mut u32; N]` for a `[Vec<u32>; N]` field.
    pub fn mut_slice_ptr_type(&self) -> TokenStream {
        match self {
            IterableField::PlainVec(plain_vec) => {
                let ty = &plain_vec.ty;
                quote! {
                    *mut [#ty]
                }
            }
            IterableField::ArrayOfVecs(array_of_vecs) => {
                let inner_type = &array_of_vecs.inner_type;
                let outer_array_size = &array_of_vecs.outer_array_size;
                quote! {
                    [*mut [#inner_type]; #outer_array_size]
                }
            }
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

    /// Generate the code to split the first element from the field.
    /// E.g. `let (head, tail) = self.field.split_at_mut(1);
    /// self.field = tail; let field = &mut (*head)[0];`
    pub fn split_first(&self) -> TokenStream {
        match self {
            IterableField::PlainVec(plain_vec) => {
                let name = &plain_vec.name;
                let head = format_ident!("{}_head", name);
                let tail = format_ident!("{}_tail", name);
                quote! {
                    let (#head, #tail) = self.#name.split_at_mut(1);
                    self.#name = #tail;
                    let #name = &mut (*(#head))[0];
                }
            }
            IterableField::ArrayOfVecs(array_of_vecs) => {
                let name = &array_of_vecs.name;
                quote! {
                    let #name = self.#name.each_mut().map(|v| {
                        let (head, tail) = v.split_at_mut(1);
                        *v = tail;
                        &mut (*head)[0]
                    });
                }
            }
        }
    }

    /// Generate the code to split the last element from the field.
    /// E.g. `let (head, tail) = self.field.split_at_mut(len - 1);
    pub fn split_last(&self, length: &Ident) -> TokenStream {
        match self {
            IterableField::PlainVec(plain_vec) => {
                let name = &plain_vec.name;
                let head = format_ident!("{}_head", name);
                let tail = format_ident!("{}_tail", name);
                quote! {
                    let (
                        #head,
                        #tail,
                    ) = self.#name.split_at_mut(#length - 1);
                    self.#name = #head;
                    let #name = &mut (*#tail)[0];
                }
            }
            IterableField::ArrayOfVecs(array_of_vecs) => {
                let name = &array_of_vecs.name;
                quote! {
                    let #name = self.#name.each_mut().map(|v| {
                        let (head, tail) = v.split_at_mut(#length - 1);
                        *v = head;
                        &mut (*tail)[0]
                    });
                }
            }
        }
    }

    /// Generate the code to get a mutable slice of the field.
    /// E.g. `self.field.as_mut_slice()`
    /// E.g. `self.field.each_mut().map(|v| v.as_mut_slice())`
    pub fn as_mut_slice(&self) -> TokenStream {
        match self {
            IterableField::PlainVec(plain_vec) => {
                let name = &plain_vec.name;
                quote! {
                    #name.as_mut_slice()
                }
            }
            IterableField::ArrayOfVecs(array_of_vecs) => {
                let name = &array_of_vecs.name;
                quote! {
                    #name.each_mut().map(|v| v.as_mut_slice())
                }
            }
        }
    }

    /// Generate the code to get a mutable pointer a mutable slice of the field.
    /// E.g. `'a mut [u32]` -> `*mut [u32]`. Achieved by casting: `as *mut _`.
    pub fn as_mut_ptr(&self) -> TokenStream {
        match self {
            IterableField::PlainVec(plain_vec) => {
                let name = &plain_vec.name;
                quote! {
                    #name as *mut _
                }
            }
            IterableField::ArrayOfVecs(array_of_vecs) => {
                let name = &array_of_vecs.name;
                quote! {
                    #name.map(|v| v as *mut _)
                }
            }
        }
    }

    /// Generate the code to get the length of the field.
    /// Length is assumed to be the same for all fields on every coordinate.
    /// E.g. `self.field.len()`
    /// E.g. `self.field[0].len()`
    pub fn get_len(&self) -> TokenStream {
        match self {
            IterableField::PlainVec(plain_vec) => {
                let name = &plain_vec.name;
                quote! {
                    #name.len()
                }
            }
            IterableField::ArrayOfVecs(array_of_vecs) => {
                let name = &array_of_vecs.name;
                quote! {
                    #name[0].len()
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

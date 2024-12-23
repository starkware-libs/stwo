use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Expr, Field, Ident, Lifetime, Type};

pub(super) enum IterableField {
    PlainVec(PlainVec),
    ArrayOfVecs(ArrayOfVecs),
}

impl IterableField {
    pub fn from_field(field: &Field) -> Self {
        match field.ty {
            Type::Array(ref outer_array) => {
                // Assert that the inner type is a Vec<T> and get T:
                let inner_type = match outer_array.elem.as_ref() {
                    Type::Path(ref type_path) => {
                        if let Some(last_segment) = type_path.path.segments.last() {
                            if last_segment.ident != "Vec" {
                                panic!("Expected Vec type");
                            }
                            if let syn::PathArguments::AngleBracketed(ref args) =
                                last_segment.arguments
                            {
                                if args.args.len() != 1 {
                                    panic!("Expected one type argument");
                                }
                                if let syn::GenericArgument::Type(inner_type) =
                                    args.args.first().unwrap()
                                {
                                    inner_type
                                } else {
                                    panic!("Expected type argument");
                                }
                            } else {
                                panic!("Expected angle-bracketed arguments");
                            }
                        } else {
                            panic!("Expected last segment");
                        }
                    }
                    _ => panic!("Expected path"),
                };
                Self::ArrayOfVecs(ArrayOfVecs {
                    name: field.ident.clone().unwrap(),
                    r#type: field.ty.clone(),
                    outer_array_size: outer_array.len.clone(),
                    inner_type: inner_type.clone(),
                })
            }
            Type::Path(ref type_path) => {
                // Assert that the type is Vec<T> and get T:
                let r#type = match type_path.path.segments.last() {
                    Some(last_segment) => {
                        if last_segment.ident != "Vec" {
                            panic!("Expected Vec type");
                        }
                        if let syn::PathArguments::AngleBracketed(ref args) = last_segment.arguments
                        {
                            if args.args.len() != 1 {
                                panic!("Expected one type argument");
                            }
                            if let syn::GenericArgument::Type(r#type) = args.args.first().unwrap() {
                                r#type
                            } else {
                                panic!("Expected type argument");
                            }
                        } else {
                            panic!("Expected angle-bracketed arguments");
                        }
                    }
                    None => panic!("Expected last segment"),
                }
                .clone();
                Self::PlainVec(PlainVec {
                    name: field.ident.clone().unwrap(),
                    r#type,
                })
            }
            _ => panic!("Expected vector or array of vectors"),
        }
    }
    pub fn name(&self) -> &Ident {
        match self {
            IterableField::PlainVec(plain_vec) => &plain_vec.name,
            IterableField::ArrayOfVecs(array_of_vecs) => &array_of_vecs.name,
        }
    }
    pub fn r#type(&self) -> &Type {
        match self {
            IterableField::PlainVec(plain_vec) => &plain_vec.r#type,
            IterableField::ArrayOfVecs(array_of_vecs) => &array_of_vecs.r#type,
        }
    }
    pub fn mut_slice_type(&self, lifetime: &Lifetime) -> TokenStream {
        match self {
            IterableField::PlainVec(plain_vec) => {
                let r#type = &plain_vec.r#type;
                quote! {
                    &#lifetime mut [#r#type]
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
    pub fn mut_chunk_type(&self, lifetime: &Lifetime) -> TokenStream {
        match self {
            IterableField::PlainVec(plain_vec) => {
                let r#type = &plain_vec.r#type;
                quote! {
                    &#lifetime mut #r#type
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
    pub fn mut_ptr_type(&self) -> TokenStream {
        match self {
            IterableField::PlainVec(plain_vec) => {
                let r#type = &plain_vec.r#type;
                quote! {
                    *mut [#r#type]
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
    pub fn uninitialized(&self) -> TokenStream {
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
    pub fn split_at(&self, index: Ident) -> TokenStream {
        match self {
            IterableField::PlainVec(plain_vec) => {
                let name = &plain_vec.name;
                let tail = format_ident!("{}_tail", name);
                quote! {
                    let (
                        #name,
                        #tail
                    ) = self.#name.split_at_mut(#index);
                }
            }
            IterableField::ArrayOfVecs(array_of_vecs) => {
                let name = &array_of_vecs.name;
                let tail = format_ident!("{}_tail", name);
                let array_size = &array_of_vecs.outer_array_size;
                quote! {
                    let (
                        mut #name,
                        mut #tail
                    ):([_; #array_size],[_; #array_size])  = unsafe { (std::mem::zeroed(), std::mem::zeroed()) };
                    self.#name.into_iter().enumerate().for_each(|(i, v)| {
                        let (head, tail) = v.split_at_mut(#index);
                        #name[i] = head;
                        #tail[i] = tail;
                    });
                }
            }
        }
    }
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

pub(super) struct PlainVec {
    pub(super) name: Ident,
    pub(super) r#type: Type,
}
pub(super) struct ArrayOfVecs {
    pub(super) name: Ident,
    pub(super) r#type: Type,
    pub(super) inner_type: Type,
    pub(super) outer_array_size: Expr,
}

use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::iterable_field::IterableField;

/// Implements an "Uninitialized" function for the struct.
/// Allocates 2^`log_size` slots for every Vector.
pub fn expand_uninitialized_impl(
    struct_name: &Ident,
    iterable_fields: &[IterableField],
) -> TokenStream {
    let (field_names, allocations): (Vec<_>, Vec<_>) = iterable_fields
        .iter()
        .map(|f| (f.name(), f.uninitialized_field_allocation()))
        .unzip();
    quote! {
        impl #struct_name {
            /// # Safety
            /// The caller must ensure that the trace is populated before being used.
            #[automatically_derived]
            pub unsafe fn uninitialized(log_size: u32) -> Self {
                let len = 1 << log_size;
                #(#allocations)*
                Self {
                    #(#field_names,)*
                }
            }
    }}
}

use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::iterable_field::IterableField;

pub fn expand_uninitialized_impl(
    struct_name: &Ident,
    iterable_fields: &[IterableField],
) -> TokenStream {
    let (field_names, allocations): (Vec<_>, Vec<_>) = iterable_fields
        .iter()
        .map(|f| (f.name(), f.uninitialized()))
        .unzip();
    quote! {
        impl #struct_name {
            /// # Safety
            /// The caller must ensure that the trace is populated before being used.
            #[allow(clippy::uninit_vec)]
            pub unsafe fn uninitialized(log_size: u32) -> Self {
                let len = 1 << log_size;
                #(#allocations)*
                Self {
                    #(#field_names,)*
                }
            }
    }}
}

wit_bindgen::generate!({
    world: "aici",
    path: "../../wit",
    additional_derives: [serde::Serialize, serde::Deserialize],
    pub_export_macro: true,
});

pub use self::{aici::abi::*, exports::aici::abi::*};

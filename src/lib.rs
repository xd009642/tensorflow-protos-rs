
#[cfg(feature="prost")]
pub mod prost_gen;

#[cfg(feature="protobuf")]
pub mod protobuf_gen;

#[cfg(all(feature="prost", not(feature="protobuf")))]
pub mod protos {
    pub use crate::prost_gen::*;
}

#[cfg(all(feature="protobuf", not(feature="prost")))]
pub mod protos {
    pub use crate::protobuf_gen::*;
}

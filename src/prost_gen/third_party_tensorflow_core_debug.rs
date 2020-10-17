/// Encapsulates per-event data related to debugging.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DebuggerEventMetadata {
    #[prost(string, tag="1")]
    pub device: std::string::String,
    #[prost(int32, tag="2")]
    pub output_slot: i32,
    #[prost(int32, tag="3")]
    pub num_chunks: i32,
    #[prost(int32, tag="4")]
    pub chunk_index: i32,
}

/// Each SnapshotRecord represents one batch of pre-processed input data. A batch
/// consists of a list of tensors that we encode as TensorProtos. This message
/// doesn't store the structure of the batch.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SnapshotRecord {
    #[prost(message, repeated, tag="1")]
    pub tensor: ::std::vec::Vec<super::super::TensorProto>,
}
/// This stores the metadata information present in each snapshot record.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SnapshotMetadataRecord {
    #[prost(string, tag="1")]
    pub graph_hash: std::string::String,
    #[prost(string, tag="2")]
    pub run_id: std::string::String,
    #[prost(int64, tag="3")]
    pub creation_timestamp: i64,
    #[prost(bool, tag="1000")]
    pub finalized: bool,
}

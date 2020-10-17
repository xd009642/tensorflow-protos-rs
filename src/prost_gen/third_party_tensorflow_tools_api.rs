#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TfapiMember {
    #[prost(string, optional, tag="1")]
    pub name: ::std::option::Option<std::string::String>,
    #[prost(string, optional, tag="2")]
    pub mtype: ::std::option::Option<std::string::String>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TfapiMethod {
    #[prost(string, optional, tag="1")]
    pub name: ::std::option::Option<std::string::String>,
    #[prost(string, optional, tag="2")]
    pub path: ::std::option::Option<std::string::String>,
    #[prost(string, optional, tag="3")]
    pub argspec: ::std::option::Option<std::string::String>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TfapiModule {
    #[prost(message, repeated, tag="1")]
    pub member: ::std::vec::Vec<TfapiMember>,
    #[prost(message, repeated, tag="2")]
    pub member_method: ::std::vec::Vec<TfapiMethod>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TfapiClass {
    #[prost(string, repeated, tag="1")]
    pub is_instance: ::std::vec::Vec<std::string::String>,
    #[prost(message, repeated, tag="2")]
    pub member: ::std::vec::Vec<TfapiMember>,
    #[prost(message, repeated, tag="3")]
    pub member_method: ::std::vec::Vec<TfapiMethod>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TfapiProto {
    #[prost(message, optional, tag="1")]
    pub descriptor: ::std::option::Option<::prost_types::DescriptorProto>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TfapiObject {
    #[prost(string, optional, tag="1")]
    pub path: ::std::option::Option<std::string::String>,
    #[prost(message, optional, tag="2")]
    pub tf_module: ::std::option::Option<TfapiModule>,
    #[prost(message, optional, tag="3")]
    pub tf_class: ::std::option::Option<TfapiClass>,
    #[prost(message, optional, tag="4")]
    pub tf_proto: ::std::option::Option<TfapiProto>,
}

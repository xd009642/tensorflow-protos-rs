// This file is generated by rust-protobuf 2.17.0. Do not edit
// @generated

// https://github.com/rust-lang/rust-clippy/issues/702
#![allow(unknown_lints)]
#![allow(clippy::all)]

#![allow(unused_attributes)]
#![rustfmt::skip]

#![allow(box_pointers)]
#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(trivial_casts)]
#![allow(unused_imports)]
#![allow(unused_results)]
//! Generated file from `tensorflow/core/protobuf/tensorflow_server.proto`

/// Generated files are compatible only with the same version
/// of protobuf runtime.
// const _PROTOBUF_VERSION_CHECK: () = ::protobuf::VERSION_2_17_0;

#[derive(PartialEq,Clone,Default)]
pub struct ServerDef {
    // message fields
    pub cluster: ::protobuf::SingularPtrField<super::cluster::ClusterDef>,
    pub job_name: ::std::string::String,
    pub task_index: i32,
    pub default_session_config: ::protobuf::SingularPtrField<super::config::ConfigProto>,
    pub protocol: ::std::string::String,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a ServerDef {
    fn default() -> &'a ServerDef {
        <ServerDef as ::protobuf::Message>::default_instance()
    }
}

impl ServerDef {
    pub fn new() -> ServerDef {
        ::std::default::Default::default()
    }

    // .tensorflow.ClusterDef cluster = 1;


    pub fn get_cluster(&self) -> &super::cluster::ClusterDef {
        self.cluster.as_ref().unwrap_or_else(|| <super::cluster::ClusterDef as ::protobuf::Message>::default_instance())
    }
    pub fn clear_cluster(&mut self) {
        self.cluster.clear();
    }

    pub fn has_cluster(&self) -> bool {
        self.cluster.is_some()
    }

    // Param is passed by value, moved
    pub fn set_cluster(&mut self, v: super::cluster::ClusterDef) {
        self.cluster = ::protobuf::SingularPtrField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_cluster(&mut self) -> &mut super::cluster::ClusterDef {
        if self.cluster.is_none() {
            self.cluster.set_default();
        }
        self.cluster.as_mut().unwrap()
    }

    // Take field
    pub fn take_cluster(&mut self) -> super::cluster::ClusterDef {
        self.cluster.take().unwrap_or_else(|| super::cluster::ClusterDef::new())
    }

    // string job_name = 2;


    pub fn get_job_name(&self) -> &str {
        &self.job_name
    }
    pub fn clear_job_name(&mut self) {
        self.job_name.clear();
    }

    // Param is passed by value, moved
    pub fn set_job_name(&mut self, v: ::std::string::String) {
        self.job_name = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_job_name(&mut self) -> &mut ::std::string::String {
        &mut self.job_name
    }

    // Take field
    pub fn take_job_name(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.job_name, ::std::string::String::new())
    }

    // int32 task_index = 3;


    pub fn get_task_index(&self) -> i32 {
        self.task_index
    }
    pub fn clear_task_index(&mut self) {
        self.task_index = 0;
    }

    // Param is passed by value, moved
    pub fn set_task_index(&mut self, v: i32) {
        self.task_index = v;
    }

    // .tensorflow.ConfigProto default_session_config = 4;


    pub fn get_default_session_config(&self) -> &super::config::ConfigProto {
        self.default_session_config.as_ref().unwrap_or_else(|| <super::config::ConfigProto as ::protobuf::Message>::default_instance())
    }
    pub fn clear_default_session_config(&mut self) {
        self.default_session_config.clear();
    }

    pub fn has_default_session_config(&self) -> bool {
        self.default_session_config.is_some()
    }

    // Param is passed by value, moved
    pub fn set_default_session_config(&mut self, v: super::config::ConfigProto) {
        self.default_session_config = ::protobuf::SingularPtrField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_default_session_config(&mut self) -> &mut super::config::ConfigProto {
        if self.default_session_config.is_none() {
            self.default_session_config.set_default();
        }
        self.default_session_config.as_mut().unwrap()
    }

    // Take field
    pub fn take_default_session_config(&mut self) -> super::config::ConfigProto {
        self.default_session_config.take().unwrap_or_else(|| super::config::ConfigProto::new())
    }

    // string protocol = 5;


    pub fn get_protocol(&self) -> &str {
        &self.protocol
    }
    pub fn clear_protocol(&mut self) {
        self.protocol.clear();
    }

    // Param is passed by value, moved
    pub fn set_protocol(&mut self, v: ::std::string::String) {
        self.protocol = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_protocol(&mut self) -> &mut ::std::string::String {
        &mut self.protocol
    }

    // Take field
    pub fn take_protocol(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.protocol, ::std::string::String::new())
    }
}

impl ::protobuf::Message for ServerDef {
    fn is_initialized(&self) -> bool {
        for v in &self.cluster {
            if !v.is_initialized() {
                return false;
            }
        };
        for v in &self.default_session_config {
            if !v.is_initialized() {
                return false;
            }
        };
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_singular_message_into(wire_type, is, &mut self.cluster)?;
                },
                2 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.job_name)?;
                },
                3 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int32()?;
                    self.task_index = tmp;
                },
                4 => {
                    ::protobuf::rt::read_singular_message_into(wire_type, is, &mut self.default_session_config)?;
                },
                5 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.protocol)?;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        if let Some(ref v) = self.cluster.as_ref() {
            let len = v.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        }
        if !self.job_name.is_empty() {
            my_size += ::protobuf::rt::string_size(2, &self.job_name);
        }
        if self.task_index != 0 {
            my_size += ::protobuf::rt::value_size(3, self.task_index, ::protobuf::wire_format::WireTypeVarint);
        }
        if let Some(ref v) = self.default_session_config.as_ref() {
            let len = v.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        }
        if !self.protocol.is_empty() {
            my_size += ::protobuf::rt::string_size(5, &self.protocol);
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        if let Some(ref v) = self.cluster.as_ref() {
            os.write_tag(1, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        }
        if !self.job_name.is_empty() {
            os.write_string(2, &self.job_name)?;
        }
        if self.task_index != 0 {
            os.write_int32(3, self.task_index)?;
        }
        if let Some(ref v) = self.default_session_config.as_ref() {
            os.write_tag(4, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        }
        if !self.protocol.is_empty() {
            os.write_string(5, &self.protocol)?;
        }
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn (::std::any::Any) {
        self as &dyn (::std::any::Any)
    }
    fn as_any_mut(&mut self) -> &mut dyn (::std::any::Any) {
        self as &mut dyn (::std::any::Any)
    }
    fn into_any(self: ::std::boxed::Box<Self>) -> ::std::boxed::Box<dyn (::std::any::Any)> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        Self::descriptor_static()
    }

    fn new() -> ServerDef {
        ServerDef::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_singular_ptr_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<super::cluster::ClusterDef>>(
                "cluster",
                |m: &ServerDef| { &m.cluster },
                |m: &mut ServerDef| { &mut m.cluster },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "job_name",
                |m: &ServerDef| { &m.job_name },
                |m: &mut ServerDef| { &mut m.job_name },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeInt32>(
                "task_index",
                |m: &ServerDef| { &m.task_index },
                |m: &mut ServerDef| { &mut m.task_index },
            ));
            fields.push(::protobuf::reflect::accessor::make_singular_ptr_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<super::config::ConfigProto>>(
                "default_session_config",
                |m: &ServerDef| { &m.default_session_config },
                |m: &mut ServerDef| { &mut m.default_session_config },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "protocol",
                |m: &ServerDef| { &m.protocol },
                |m: &mut ServerDef| { &mut m.protocol },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<ServerDef>(
                "ServerDef",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static ServerDef {
        static instance: ::protobuf::rt::LazyV2<ServerDef> = ::protobuf::rt::LazyV2::INIT;
        instance.get(ServerDef::new)
    }
}

impl ::protobuf::Clear for ServerDef {
    fn clear(&mut self) {
        self.cluster.clear();
        self.job_name.clear();
        self.task_index = 0;
        self.default_session_config.clear();
        self.protocol.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for ServerDef {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for ServerDef {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

static file_descriptor_proto_data: &'static [u8] = b"\
    \n0tensorflow/core/protobuf/tensorflow_server.proto\x12\ntensorflow\x1a%\
    tensorflow/core/protobuf/config.proto\x1a&tensorflow/core/protobuf/clust\
    er.proto\"\xe2\x01\n\tServerDef\x120\n\x07cluster\x18\x01\x20\x01(\x0b2\
    \x16.tensorflow.ClusterDefR\x07cluster\x12\x19\n\x08job_name\x18\x02\x20\
    \x01(\tR\x07jobName\x12\x1d\n\ntask_index\x18\x03\x20\x01(\x05R\ttaskInd\
    ex\x12M\n\x16default_session_config\x18\x04\x20\x01(\x0b2\x17.tensorflow\
    .ConfigProtoR\x14defaultSessionConfig\x12\x1a\n\x08protocol\x18\x05\x20\
    \x01(\tR\x08protocolBm\n\x1aorg.tensorflow.distruntimeB\x0cServerProtosP\
    \x01Z<github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf\xf8\
    \x01\x01b\x06proto3\
";

static file_descriptor_proto_lazy: ::protobuf::rt::LazyV2<::protobuf::descriptor::FileDescriptorProto> = ::protobuf::rt::LazyV2::INIT;

fn parse_descriptor_proto() -> ::protobuf::descriptor::FileDescriptorProto {
    ::protobuf::parse_from_bytes(file_descriptor_proto_data).unwrap()
}

pub fn file_descriptor_proto() -> &'static ::protobuf::descriptor::FileDescriptorProto {
    file_descriptor_proto_lazy.get(|| {
        parse_descriptor_proto()
    })
}
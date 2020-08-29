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
//! Generated file from `tensorflow/compiler/tf2xla/host_compute_metadata.proto`

/// Generated files are compatible only with the same version
/// of protobuf runtime.
// const _PROTOBUF_VERSION_CHECK: () = ::protobuf::VERSION_2_17_0;

#[derive(PartialEq,Clone,Default)]
pub struct TensorMetadata {
    // message fields
    pub field_type: super::types::DataType,
    pub shape: ::protobuf::SingularPtrField<super::tensor_shape::TensorShapeProto>,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a TensorMetadata {
    fn default() -> &'a TensorMetadata {
        <TensorMetadata as ::protobuf::Message>::default_instance()
    }
}

impl TensorMetadata {
    pub fn new() -> TensorMetadata {
        ::std::default::Default::default()
    }

    // .tensorflow.DataType type = 1;


    pub fn get_field_type(&self) -> super::types::DataType {
        self.field_type
    }
    pub fn clear_field_type(&mut self) {
        self.field_type = super::types::DataType::DT_INVALID;
    }

    // Param is passed by value, moved
    pub fn set_field_type(&mut self, v: super::types::DataType) {
        self.field_type = v;
    }

    // .tensorflow.TensorShapeProto shape = 2;


    pub fn get_shape(&self) -> &super::tensor_shape::TensorShapeProto {
        self.shape.as_ref().unwrap_or_else(|| <super::tensor_shape::TensorShapeProto as ::protobuf::Message>::default_instance())
    }
    pub fn clear_shape(&mut self) {
        self.shape.clear();
    }

    pub fn has_shape(&self) -> bool {
        self.shape.is_some()
    }

    // Param is passed by value, moved
    pub fn set_shape(&mut self, v: super::tensor_shape::TensorShapeProto) {
        self.shape = ::protobuf::SingularPtrField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_shape(&mut self) -> &mut super::tensor_shape::TensorShapeProto {
        if self.shape.is_none() {
            self.shape.set_default();
        }
        self.shape.as_mut().unwrap()
    }

    // Take field
    pub fn take_shape(&mut self) -> super::tensor_shape::TensorShapeProto {
        self.shape.take().unwrap_or_else(|| super::tensor_shape::TensorShapeProto::new())
    }
}

impl ::protobuf::Message for TensorMetadata {
    fn is_initialized(&self) -> bool {
        for v in &self.shape {
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
                    ::protobuf::rt::read_proto3_enum_with_unknown_fields_into(wire_type, is, &mut self.field_type, 1, &mut self.unknown_fields)?
                },
                2 => {
                    ::protobuf::rt::read_singular_message_into(wire_type, is, &mut self.shape)?;
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
        if self.field_type != super::types::DataType::DT_INVALID {
            my_size += ::protobuf::rt::enum_size(1, self.field_type);
        }
        if let Some(ref v) = self.shape.as_ref() {
            let len = v.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        if self.field_type != super::types::DataType::DT_INVALID {
            os.write_enum(1, ::protobuf::ProtobufEnum::value(&self.field_type))?;
        }
        if let Some(ref v) = self.shape.as_ref() {
            os.write_tag(2, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
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

    fn new() -> TensorMetadata {
        TensorMetadata::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeEnum<super::types::DataType>>(
                "type",
                |m: &TensorMetadata| { &m.field_type },
                |m: &mut TensorMetadata| { &mut m.field_type },
            ));
            fields.push(::protobuf::reflect::accessor::make_singular_ptr_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<super::tensor_shape::TensorShapeProto>>(
                "shape",
                |m: &TensorMetadata| { &m.shape },
                |m: &mut TensorMetadata| { &mut m.shape },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<TensorMetadata>(
                "TensorMetadata",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static TensorMetadata {
        static instance: ::protobuf::rt::LazyV2<TensorMetadata> = ::protobuf::rt::LazyV2::INIT;
        instance.get(TensorMetadata::new)
    }
}

impl ::protobuf::Clear for TensorMetadata {
    fn clear(&mut self) {
        self.field_type = super::types::DataType::DT_INVALID;
        self.shape.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for TensorMetadata {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for TensorMetadata {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct HostTransferMetadata {
    // message fields
    pub key: ::std::string::String,
    pub metadata: ::protobuf::RepeatedField<TensorMetadata>,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a HostTransferMetadata {
    fn default() -> &'a HostTransferMetadata {
        <HostTransferMetadata as ::protobuf::Message>::default_instance()
    }
}

impl HostTransferMetadata {
    pub fn new() -> HostTransferMetadata {
        ::std::default::Default::default()
    }

    // string key = 1;


    pub fn get_key(&self) -> &str {
        &self.key
    }
    pub fn clear_key(&mut self) {
        self.key.clear();
    }

    // Param is passed by value, moved
    pub fn set_key(&mut self, v: ::std::string::String) {
        self.key = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_key(&mut self) -> &mut ::std::string::String {
        &mut self.key
    }

    // Take field
    pub fn take_key(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.key, ::std::string::String::new())
    }

    // repeated .tensorflow.tf2xla.TensorMetadata metadata = 2;


    pub fn get_metadata(&self) -> &[TensorMetadata] {
        &self.metadata
    }
    pub fn clear_metadata(&mut self) {
        self.metadata.clear();
    }

    // Param is passed by value, moved
    pub fn set_metadata(&mut self, v: ::protobuf::RepeatedField<TensorMetadata>) {
        self.metadata = v;
    }

    // Mutable pointer to the field.
    pub fn mut_metadata(&mut self) -> &mut ::protobuf::RepeatedField<TensorMetadata> {
        &mut self.metadata
    }

    // Take field
    pub fn take_metadata(&mut self) -> ::protobuf::RepeatedField<TensorMetadata> {
        ::std::mem::replace(&mut self.metadata, ::protobuf::RepeatedField::new())
    }
}

impl ::protobuf::Message for HostTransferMetadata {
    fn is_initialized(&self) -> bool {
        for v in &self.metadata {
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
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.key)?;
                },
                2 => {
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.metadata)?;
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
        if !self.key.is_empty() {
            my_size += ::protobuf::rt::string_size(1, &self.key);
        }
        for value in &self.metadata {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        if !self.key.is_empty() {
            os.write_string(1, &self.key)?;
        }
        for v in &self.metadata {
            os.write_tag(2, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        };
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

    fn new() -> HostTransferMetadata {
        HostTransferMetadata::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "key",
                |m: &HostTransferMetadata| { &m.key },
                |m: &mut HostTransferMetadata| { &mut m.key },
            ));
            fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<TensorMetadata>>(
                "metadata",
                |m: &HostTransferMetadata| { &m.metadata },
                |m: &mut HostTransferMetadata| { &mut m.metadata },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<HostTransferMetadata>(
                "HostTransferMetadata",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static HostTransferMetadata {
        static instance: ::protobuf::rt::LazyV2<HostTransferMetadata> = ::protobuf::rt::LazyV2::INIT;
        instance.get(HostTransferMetadata::new)
    }
}

impl ::protobuf::Clear for HostTransferMetadata {
    fn clear(&mut self) {
        self.key.clear();
        self.metadata.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for HostTransferMetadata {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for HostTransferMetadata {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct HostComputeMetadata {
    // message fields
    pub device_to_host: ::protobuf::RepeatedField<HostTransferMetadata>,
    pub host_to_device: ::protobuf::RepeatedField<HostTransferMetadata>,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a HostComputeMetadata {
    fn default() -> &'a HostComputeMetadata {
        <HostComputeMetadata as ::protobuf::Message>::default_instance()
    }
}

impl HostComputeMetadata {
    pub fn new() -> HostComputeMetadata {
        ::std::default::Default::default()
    }

    // repeated .tensorflow.tf2xla.HostTransferMetadata device_to_host = 1;


    pub fn get_device_to_host(&self) -> &[HostTransferMetadata] {
        &self.device_to_host
    }
    pub fn clear_device_to_host(&mut self) {
        self.device_to_host.clear();
    }

    // Param is passed by value, moved
    pub fn set_device_to_host(&mut self, v: ::protobuf::RepeatedField<HostTransferMetadata>) {
        self.device_to_host = v;
    }

    // Mutable pointer to the field.
    pub fn mut_device_to_host(&mut self) -> &mut ::protobuf::RepeatedField<HostTransferMetadata> {
        &mut self.device_to_host
    }

    // Take field
    pub fn take_device_to_host(&mut self) -> ::protobuf::RepeatedField<HostTransferMetadata> {
        ::std::mem::replace(&mut self.device_to_host, ::protobuf::RepeatedField::new())
    }

    // repeated .tensorflow.tf2xla.HostTransferMetadata host_to_device = 2;


    pub fn get_host_to_device(&self) -> &[HostTransferMetadata] {
        &self.host_to_device
    }
    pub fn clear_host_to_device(&mut self) {
        self.host_to_device.clear();
    }

    // Param is passed by value, moved
    pub fn set_host_to_device(&mut self, v: ::protobuf::RepeatedField<HostTransferMetadata>) {
        self.host_to_device = v;
    }

    // Mutable pointer to the field.
    pub fn mut_host_to_device(&mut self) -> &mut ::protobuf::RepeatedField<HostTransferMetadata> {
        &mut self.host_to_device
    }

    // Take field
    pub fn take_host_to_device(&mut self) -> ::protobuf::RepeatedField<HostTransferMetadata> {
        ::std::mem::replace(&mut self.host_to_device, ::protobuf::RepeatedField::new())
    }
}

impl ::protobuf::Message for HostComputeMetadata {
    fn is_initialized(&self) -> bool {
        for v in &self.device_to_host {
            if !v.is_initialized() {
                return false;
            }
        };
        for v in &self.host_to_device {
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
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.device_to_host)?;
                },
                2 => {
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.host_to_device)?;
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
        for value in &self.device_to_host {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        for value in &self.host_to_device {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        for v in &self.device_to_host {
            os.write_tag(1, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        };
        for v in &self.host_to_device {
            os.write_tag(2, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        };
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

    fn new() -> HostComputeMetadata {
        HostComputeMetadata::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<HostTransferMetadata>>(
                "device_to_host",
                |m: &HostComputeMetadata| { &m.device_to_host },
                |m: &mut HostComputeMetadata| { &mut m.device_to_host },
            ));
            fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<HostTransferMetadata>>(
                "host_to_device",
                |m: &HostComputeMetadata| { &m.host_to_device },
                |m: &mut HostComputeMetadata| { &mut m.host_to_device },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<HostComputeMetadata>(
                "HostComputeMetadata",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static HostComputeMetadata {
        static instance: ::protobuf::rt::LazyV2<HostComputeMetadata> = ::protobuf::rt::LazyV2::INIT;
        instance.get(HostComputeMetadata::new)
    }
}

impl ::protobuf::Clear for HostComputeMetadata {
    fn clear(&mut self) {
        self.device_to_host.clear();
        self.host_to_device.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for HostComputeMetadata {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for HostComputeMetadata {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

static file_descriptor_proto_data: &'static [u8] = b"\
    \n6tensorflow/compiler/tf2xla/host_compute_metadata.proto\x12\x11tensorf\
    low.tf2xla\x1a,tensorflow/core/framework/tensor_shape.proto\x1a%tensorfl\
    ow/core/framework/types.proto\"n\n\x0eTensorMetadata\x12(\n\x04type\x18\
    \x01\x20\x01(\x0e2\x14.tensorflow.DataTypeR\x04type\x122\n\x05shape\x18\
    \x02\x20\x01(\x0b2\x1c.tensorflow.TensorShapeProtoR\x05shape\"g\n\x14Hos\
    tTransferMetadata\x12\x10\n\x03key\x18\x01\x20\x01(\tR\x03key\x12=\n\x08\
    metadata\x18\x02\x20\x03(\x0b2!.tensorflow.tf2xla.TensorMetadataR\x08met\
    adata\"\xb3\x01\n\x13HostComputeMetadata\x12M\n\x0edevice_to_host\x18\
    \x01\x20\x03(\x0b2'.tensorflow.tf2xla.HostTransferMetadataR\x0cdeviceToH\
    ost\x12M\n\x0ehost_to_device\x18\x02\x20\x03(\x0b2'.tensorflow.tf2xla.Ho\
    stTransferMetadataR\x0chostToDeviceB*\n\x15org.tensorflow.tf2xlaB\x0cTf2\
    XlaProtosP\x01\xf8\x01\x01b\x06proto3\
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
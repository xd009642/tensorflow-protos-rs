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
//! Generated file from `tensorflow/core/framework/resource_handle.proto`

/// Generated files are compatible only with the same version
/// of protobuf runtime.
// const _PROTOBUF_VERSION_CHECK: () = ::protobuf::VERSION_2_17_0;

#[derive(PartialEq,Clone,Default)]
pub struct ResourceHandleProto {
    // message fields
    pub device: ::std::string::String,
    pub container: ::std::string::String,
    pub name: ::std::string::String,
    pub hash_code: u64,
    pub maybe_type_name: ::std::string::String,
    pub dtypes_and_shapes: ::protobuf::RepeatedField<ResourceHandleProto_DtypeAndShape>,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a ResourceHandleProto {
    fn default() -> &'a ResourceHandleProto {
        <ResourceHandleProto as ::protobuf::Message>::default_instance()
    }
}

impl ResourceHandleProto {
    pub fn new() -> ResourceHandleProto {
        ::std::default::Default::default()
    }

    // string device = 1;


    pub fn get_device(&self) -> &str {
        &self.device
    }
    pub fn clear_device(&mut self) {
        self.device.clear();
    }

    // Param is passed by value, moved
    pub fn set_device(&mut self, v: ::std::string::String) {
        self.device = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_device(&mut self) -> &mut ::std::string::String {
        &mut self.device
    }

    // Take field
    pub fn take_device(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.device, ::std::string::String::new())
    }

    // string container = 2;


    pub fn get_container(&self) -> &str {
        &self.container
    }
    pub fn clear_container(&mut self) {
        self.container.clear();
    }

    // Param is passed by value, moved
    pub fn set_container(&mut self, v: ::std::string::String) {
        self.container = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_container(&mut self) -> &mut ::std::string::String {
        &mut self.container
    }

    // Take field
    pub fn take_container(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.container, ::std::string::String::new())
    }

    // string name = 3;


    pub fn get_name(&self) -> &str {
        &self.name
    }
    pub fn clear_name(&mut self) {
        self.name.clear();
    }

    // Param is passed by value, moved
    pub fn set_name(&mut self, v: ::std::string::String) {
        self.name = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_name(&mut self) -> &mut ::std::string::String {
        &mut self.name
    }

    // Take field
    pub fn take_name(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.name, ::std::string::String::new())
    }

    // uint64 hash_code = 4;


    pub fn get_hash_code(&self) -> u64 {
        self.hash_code
    }
    pub fn clear_hash_code(&mut self) {
        self.hash_code = 0;
    }

    // Param is passed by value, moved
    pub fn set_hash_code(&mut self, v: u64) {
        self.hash_code = v;
    }

    // string maybe_type_name = 5;


    pub fn get_maybe_type_name(&self) -> &str {
        &self.maybe_type_name
    }
    pub fn clear_maybe_type_name(&mut self) {
        self.maybe_type_name.clear();
    }

    // Param is passed by value, moved
    pub fn set_maybe_type_name(&mut self, v: ::std::string::String) {
        self.maybe_type_name = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_maybe_type_name(&mut self) -> &mut ::std::string::String {
        &mut self.maybe_type_name
    }

    // Take field
    pub fn take_maybe_type_name(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.maybe_type_name, ::std::string::String::new())
    }

    // repeated .tensorflow.ResourceHandleProto.DtypeAndShape dtypes_and_shapes = 6;


    pub fn get_dtypes_and_shapes(&self) -> &[ResourceHandleProto_DtypeAndShape] {
        &self.dtypes_and_shapes
    }
    pub fn clear_dtypes_and_shapes(&mut self) {
        self.dtypes_and_shapes.clear();
    }

    // Param is passed by value, moved
    pub fn set_dtypes_and_shapes(&mut self, v: ::protobuf::RepeatedField<ResourceHandleProto_DtypeAndShape>) {
        self.dtypes_and_shapes = v;
    }

    // Mutable pointer to the field.
    pub fn mut_dtypes_and_shapes(&mut self) -> &mut ::protobuf::RepeatedField<ResourceHandleProto_DtypeAndShape> {
        &mut self.dtypes_and_shapes
    }

    // Take field
    pub fn take_dtypes_and_shapes(&mut self) -> ::protobuf::RepeatedField<ResourceHandleProto_DtypeAndShape> {
        ::std::mem::replace(&mut self.dtypes_and_shapes, ::protobuf::RepeatedField::new())
    }
}

impl ::protobuf::Message for ResourceHandleProto {
    fn is_initialized(&self) -> bool {
        for v in &self.dtypes_and_shapes {
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
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.device)?;
                },
                2 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.container)?;
                },
                3 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.name)?;
                },
                4 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_uint64()?;
                    self.hash_code = tmp;
                },
                5 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.maybe_type_name)?;
                },
                6 => {
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.dtypes_and_shapes)?;
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
        if !self.device.is_empty() {
            my_size += ::protobuf::rt::string_size(1, &self.device);
        }
        if !self.container.is_empty() {
            my_size += ::protobuf::rt::string_size(2, &self.container);
        }
        if !self.name.is_empty() {
            my_size += ::protobuf::rt::string_size(3, &self.name);
        }
        if self.hash_code != 0 {
            my_size += ::protobuf::rt::value_size(4, self.hash_code, ::protobuf::wire_format::WireTypeVarint);
        }
        if !self.maybe_type_name.is_empty() {
            my_size += ::protobuf::rt::string_size(5, &self.maybe_type_name);
        }
        for value in &self.dtypes_and_shapes {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        if !self.device.is_empty() {
            os.write_string(1, &self.device)?;
        }
        if !self.container.is_empty() {
            os.write_string(2, &self.container)?;
        }
        if !self.name.is_empty() {
            os.write_string(3, &self.name)?;
        }
        if self.hash_code != 0 {
            os.write_uint64(4, self.hash_code)?;
        }
        if !self.maybe_type_name.is_empty() {
            os.write_string(5, &self.maybe_type_name)?;
        }
        for v in &self.dtypes_and_shapes {
            os.write_tag(6, ::protobuf::wire_format::WireTypeLengthDelimited)?;
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

    fn new() -> ResourceHandleProto {
        ResourceHandleProto::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "device",
                |m: &ResourceHandleProto| { &m.device },
                |m: &mut ResourceHandleProto| { &mut m.device },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "container",
                |m: &ResourceHandleProto| { &m.container },
                |m: &mut ResourceHandleProto| { &mut m.container },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "name",
                |m: &ResourceHandleProto| { &m.name },
                |m: &mut ResourceHandleProto| { &mut m.name },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeUint64>(
                "hash_code",
                |m: &ResourceHandleProto| { &m.hash_code },
                |m: &mut ResourceHandleProto| { &mut m.hash_code },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "maybe_type_name",
                |m: &ResourceHandleProto| { &m.maybe_type_name },
                |m: &mut ResourceHandleProto| { &mut m.maybe_type_name },
            ));
            fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<ResourceHandleProto_DtypeAndShape>>(
                "dtypes_and_shapes",
                |m: &ResourceHandleProto| { &m.dtypes_and_shapes },
                |m: &mut ResourceHandleProto| { &mut m.dtypes_and_shapes },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<ResourceHandleProto>(
                "ResourceHandleProto",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static ResourceHandleProto {
        static instance: ::protobuf::rt::LazyV2<ResourceHandleProto> = ::protobuf::rt::LazyV2::INIT;
        instance.get(ResourceHandleProto::new)
    }
}

impl ::protobuf::Clear for ResourceHandleProto {
    fn clear(&mut self) {
        self.device.clear();
        self.container.clear();
        self.name.clear();
        self.hash_code = 0;
        self.maybe_type_name.clear();
        self.dtypes_and_shapes.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for ResourceHandleProto {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for ResourceHandleProto {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct ResourceHandleProto_DtypeAndShape {
    // message fields
    pub dtype: super::types::DataType,
    pub shape: ::protobuf::SingularPtrField<super::tensor_shape::TensorShapeProto>,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a ResourceHandleProto_DtypeAndShape {
    fn default() -> &'a ResourceHandleProto_DtypeAndShape {
        <ResourceHandleProto_DtypeAndShape as ::protobuf::Message>::default_instance()
    }
}

impl ResourceHandleProto_DtypeAndShape {
    pub fn new() -> ResourceHandleProto_DtypeAndShape {
        ::std::default::Default::default()
    }

    // .tensorflow.DataType dtype = 1;


    pub fn get_dtype(&self) -> super::types::DataType {
        self.dtype
    }
    pub fn clear_dtype(&mut self) {
        self.dtype = super::types::DataType::DT_INVALID;
    }

    // Param is passed by value, moved
    pub fn set_dtype(&mut self, v: super::types::DataType) {
        self.dtype = v;
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

impl ::protobuf::Message for ResourceHandleProto_DtypeAndShape {
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
                    ::protobuf::rt::read_proto3_enum_with_unknown_fields_into(wire_type, is, &mut self.dtype, 1, &mut self.unknown_fields)?
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
        if self.dtype != super::types::DataType::DT_INVALID {
            my_size += ::protobuf::rt::enum_size(1, self.dtype);
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
        if self.dtype != super::types::DataType::DT_INVALID {
            os.write_enum(1, ::protobuf::ProtobufEnum::value(&self.dtype))?;
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

    fn new() -> ResourceHandleProto_DtypeAndShape {
        ResourceHandleProto_DtypeAndShape::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeEnum<super::types::DataType>>(
                "dtype",
                |m: &ResourceHandleProto_DtypeAndShape| { &m.dtype },
                |m: &mut ResourceHandleProto_DtypeAndShape| { &mut m.dtype },
            ));
            fields.push(::protobuf::reflect::accessor::make_singular_ptr_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<super::tensor_shape::TensorShapeProto>>(
                "shape",
                |m: &ResourceHandleProto_DtypeAndShape| { &m.shape },
                |m: &mut ResourceHandleProto_DtypeAndShape| { &mut m.shape },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<ResourceHandleProto_DtypeAndShape>(
                "ResourceHandleProto.DtypeAndShape",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static ResourceHandleProto_DtypeAndShape {
        static instance: ::protobuf::rt::LazyV2<ResourceHandleProto_DtypeAndShape> = ::protobuf::rt::LazyV2::INIT;
        instance.get(ResourceHandleProto_DtypeAndShape::new)
    }
}

impl ::protobuf::Clear for ResourceHandleProto_DtypeAndShape {
    fn clear(&mut self) {
        self.dtype = super::types::DataType::DT_INVALID;
        self.shape.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for ResourceHandleProto_DtypeAndShape {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for ResourceHandleProto_DtypeAndShape {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

static file_descriptor_proto_data: &'static [u8] = b"\
    \n/tensorflow/core/framework/resource_handle.proto\x12\ntensorflow\x1a,t\
    ensorflow/core/framework/tensor_shape.proto\x1a%tensorflow/core/framewor\
    k/types.proto\"\xf0\x02\n\x13ResourceHandleProto\x12\x16\n\x06device\x18\
    \x01\x20\x01(\tR\x06device\x12\x1c\n\tcontainer\x18\x02\x20\x01(\tR\tcon\
    tainer\x12\x12\n\x04name\x18\x03\x20\x01(\tR\x04name\x12\x1b\n\thash_cod\
    e\x18\x04\x20\x01(\x04R\x08hashCode\x12&\n\x0fmaybe_type_name\x18\x05\
    \x20\x01(\tR\rmaybeTypeName\x12Y\n\x11dtypes_and_shapes\x18\x06\x20\x03(\
    \x0b2-.tensorflow.ResourceHandleProto.DtypeAndShapeR\x0fdtypesAndShapes\
    \x1ao\n\rDtypeAndShape\x12*\n\x05dtype\x18\x01\x20\x01(\x0e2\x14.tensorf\
    low.DataTypeR\x05dtype\x122\n\x05shape\x18\x02\x20\x01(\x0b2\x1c.tensorf\
    low.TensorShapeProtoR\x05shapeBn\n\x18org.tensorflow.frameworkB\x0eResou\
    rceHandleP\x01Z=github.com/tensorflow/tensorflow/tensorflow/go/core/fram\
    ework\xf8\x01\x01b\x06proto3\
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

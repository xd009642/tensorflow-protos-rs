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
//! Generated file from `tensorflow/core/protobuf/saver.proto`

/// Generated files are compatible only with the same version
/// of protobuf runtime.
// const _PROTOBUF_VERSION_CHECK: () = ::protobuf::VERSION_2_17_0;

#[derive(PartialEq,Clone,Default)]
pub struct SaverDef {
    // message fields
    pub filename_tensor_name: ::std::string::String,
    pub save_tensor_name: ::std::string::String,
    pub restore_op_name: ::std::string::String,
    pub max_to_keep: i32,
    pub sharded: bool,
    pub keep_checkpoint_every_n_hours: f32,
    pub version: SaverDef_CheckpointFormatVersion,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a SaverDef {
    fn default() -> &'a SaverDef {
        <SaverDef as ::protobuf::Message>::default_instance()
    }
}

impl SaverDef {
    pub fn new() -> SaverDef {
        ::std::default::Default::default()
    }

    // string filename_tensor_name = 1;


    pub fn get_filename_tensor_name(&self) -> &str {
        &self.filename_tensor_name
    }
    pub fn clear_filename_tensor_name(&mut self) {
        self.filename_tensor_name.clear();
    }

    // Param is passed by value, moved
    pub fn set_filename_tensor_name(&mut self, v: ::std::string::String) {
        self.filename_tensor_name = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_filename_tensor_name(&mut self) -> &mut ::std::string::String {
        &mut self.filename_tensor_name
    }

    // Take field
    pub fn take_filename_tensor_name(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.filename_tensor_name, ::std::string::String::new())
    }

    // string save_tensor_name = 2;


    pub fn get_save_tensor_name(&self) -> &str {
        &self.save_tensor_name
    }
    pub fn clear_save_tensor_name(&mut self) {
        self.save_tensor_name.clear();
    }

    // Param is passed by value, moved
    pub fn set_save_tensor_name(&mut self, v: ::std::string::String) {
        self.save_tensor_name = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_save_tensor_name(&mut self) -> &mut ::std::string::String {
        &mut self.save_tensor_name
    }

    // Take field
    pub fn take_save_tensor_name(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.save_tensor_name, ::std::string::String::new())
    }

    // string restore_op_name = 3;


    pub fn get_restore_op_name(&self) -> &str {
        &self.restore_op_name
    }
    pub fn clear_restore_op_name(&mut self) {
        self.restore_op_name.clear();
    }

    // Param is passed by value, moved
    pub fn set_restore_op_name(&mut self, v: ::std::string::String) {
        self.restore_op_name = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_restore_op_name(&mut self) -> &mut ::std::string::String {
        &mut self.restore_op_name
    }

    // Take field
    pub fn take_restore_op_name(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.restore_op_name, ::std::string::String::new())
    }

    // int32 max_to_keep = 4;


    pub fn get_max_to_keep(&self) -> i32 {
        self.max_to_keep
    }
    pub fn clear_max_to_keep(&mut self) {
        self.max_to_keep = 0;
    }

    // Param is passed by value, moved
    pub fn set_max_to_keep(&mut self, v: i32) {
        self.max_to_keep = v;
    }

    // bool sharded = 5;


    pub fn get_sharded(&self) -> bool {
        self.sharded
    }
    pub fn clear_sharded(&mut self) {
        self.sharded = false;
    }

    // Param is passed by value, moved
    pub fn set_sharded(&mut self, v: bool) {
        self.sharded = v;
    }

    // float keep_checkpoint_every_n_hours = 6;


    pub fn get_keep_checkpoint_every_n_hours(&self) -> f32 {
        self.keep_checkpoint_every_n_hours
    }
    pub fn clear_keep_checkpoint_every_n_hours(&mut self) {
        self.keep_checkpoint_every_n_hours = 0.;
    }

    // Param is passed by value, moved
    pub fn set_keep_checkpoint_every_n_hours(&mut self, v: f32) {
        self.keep_checkpoint_every_n_hours = v;
    }

    // .tensorflow.SaverDef.CheckpointFormatVersion version = 7;


    pub fn get_version(&self) -> SaverDef_CheckpointFormatVersion {
        self.version
    }
    pub fn clear_version(&mut self) {
        self.version = SaverDef_CheckpointFormatVersion::LEGACY;
    }

    // Param is passed by value, moved
    pub fn set_version(&mut self, v: SaverDef_CheckpointFormatVersion) {
        self.version = v;
    }
}

impl ::protobuf::Message for SaverDef {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.filename_tensor_name)?;
                },
                2 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.save_tensor_name)?;
                },
                3 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.restore_op_name)?;
                },
                4 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int32()?;
                    self.max_to_keep = tmp;
                },
                5 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_bool()?;
                    self.sharded = tmp;
                },
                6 => {
                    if wire_type != ::protobuf::wire_format::WireTypeFixed32 {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_float()?;
                    self.keep_checkpoint_every_n_hours = tmp;
                },
                7 => {
                    ::protobuf::rt::read_proto3_enum_with_unknown_fields_into(wire_type, is, &mut self.version, 7, &mut self.unknown_fields)?
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
        if !self.filename_tensor_name.is_empty() {
            my_size += ::protobuf::rt::string_size(1, &self.filename_tensor_name);
        }
        if !self.save_tensor_name.is_empty() {
            my_size += ::protobuf::rt::string_size(2, &self.save_tensor_name);
        }
        if !self.restore_op_name.is_empty() {
            my_size += ::protobuf::rt::string_size(3, &self.restore_op_name);
        }
        if self.max_to_keep != 0 {
            my_size += ::protobuf::rt::value_size(4, self.max_to_keep, ::protobuf::wire_format::WireTypeVarint);
        }
        if self.sharded != false {
            my_size += 2;
        }
        if self.keep_checkpoint_every_n_hours != 0. {
            my_size += 5;
        }
        if self.version != SaverDef_CheckpointFormatVersion::LEGACY {
            my_size += ::protobuf::rt::enum_size(7, self.version);
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        if !self.filename_tensor_name.is_empty() {
            os.write_string(1, &self.filename_tensor_name)?;
        }
        if !self.save_tensor_name.is_empty() {
            os.write_string(2, &self.save_tensor_name)?;
        }
        if !self.restore_op_name.is_empty() {
            os.write_string(3, &self.restore_op_name)?;
        }
        if self.max_to_keep != 0 {
            os.write_int32(4, self.max_to_keep)?;
        }
        if self.sharded != false {
            os.write_bool(5, self.sharded)?;
        }
        if self.keep_checkpoint_every_n_hours != 0. {
            os.write_float(6, self.keep_checkpoint_every_n_hours)?;
        }
        if self.version != SaverDef_CheckpointFormatVersion::LEGACY {
            os.write_enum(7, ::protobuf::ProtobufEnum::value(&self.version))?;
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

    fn new() -> SaverDef {
        SaverDef::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "filename_tensor_name",
                |m: &SaverDef| { &m.filename_tensor_name },
                |m: &mut SaverDef| { &mut m.filename_tensor_name },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "save_tensor_name",
                |m: &SaverDef| { &m.save_tensor_name },
                |m: &mut SaverDef| { &mut m.save_tensor_name },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "restore_op_name",
                |m: &SaverDef| { &m.restore_op_name },
                |m: &mut SaverDef| { &mut m.restore_op_name },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeInt32>(
                "max_to_keep",
                |m: &SaverDef| { &m.max_to_keep },
                |m: &mut SaverDef| { &mut m.max_to_keep },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeBool>(
                "sharded",
                |m: &SaverDef| { &m.sharded },
                |m: &mut SaverDef| { &mut m.sharded },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeFloat>(
                "keep_checkpoint_every_n_hours",
                |m: &SaverDef| { &m.keep_checkpoint_every_n_hours },
                |m: &mut SaverDef| { &mut m.keep_checkpoint_every_n_hours },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeEnum<SaverDef_CheckpointFormatVersion>>(
                "version",
                |m: &SaverDef| { &m.version },
                |m: &mut SaverDef| { &mut m.version },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<SaverDef>(
                "SaverDef",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static SaverDef {
        static instance: ::protobuf::rt::LazyV2<SaverDef> = ::protobuf::rt::LazyV2::INIT;
        instance.get(SaverDef::new)
    }
}

impl ::protobuf::Clear for SaverDef {
    fn clear(&mut self) {
        self.filename_tensor_name.clear();
        self.save_tensor_name.clear();
        self.restore_op_name.clear();
        self.max_to_keep = 0;
        self.sharded = false;
        self.keep_checkpoint_every_n_hours = 0.;
        self.version = SaverDef_CheckpointFormatVersion::LEGACY;
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for SaverDef {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for SaverDef {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

#[derive(Clone,PartialEq,Eq,Debug,Hash)]
pub enum SaverDef_CheckpointFormatVersion {
    LEGACY = 0,
    V1 = 1,
    V2 = 2,
}

impl ::protobuf::ProtobufEnum for SaverDef_CheckpointFormatVersion {
    fn value(&self) -> i32 {
        *self as i32
    }

    fn from_i32(value: i32) -> ::std::option::Option<SaverDef_CheckpointFormatVersion> {
        match value {
            0 => ::std::option::Option::Some(SaverDef_CheckpointFormatVersion::LEGACY),
            1 => ::std::option::Option::Some(SaverDef_CheckpointFormatVersion::V1),
            2 => ::std::option::Option::Some(SaverDef_CheckpointFormatVersion::V2),
            _ => ::std::option::Option::None
        }
    }

    fn values() -> &'static [Self] {
        static values: &'static [SaverDef_CheckpointFormatVersion] = &[
            SaverDef_CheckpointFormatVersion::LEGACY,
            SaverDef_CheckpointFormatVersion::V1,
            SaverDef_CheckpointFormatVersion::V2,
        ];
        values
    }

    fn enum_descriptor_static() -> &'static ::protobuf::reflect::EnumDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::EnumDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            ::protobuf::reflect::EnumDescriptor::new_pb_name::<SaverDef_CheckpointFormatVersion>("SaverDef.CheckpointFormatVersion", file_descriptor_proto())
        })
    }
}

impl ::std::marker::Copy for SaverDef_CheckpointFormatVersion {
}

impl ::std::default::Default for SaverDef_CheckpointFormatVersion {
    fn default() -> Self {
        SaverDef_CheckpointFormatVersion::LEGACY
    }
}

impl ::protobuf::reflect::ProtobufValue for SaverDef_CheckpointFormatVersion {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Enum(::protobuf::ProtobufEnum::descriptor(self))
    }
}

static file_descriptor_proto_data: &'static [u8] = b"\
    \n$tensorflow/core/protobuf/saver.proto\x12\ntensorflow\"\x89\x03\n\x08S\
    averDef\x120\n\x14filename_tensor_name\x18\x01\x20\x01(\tR\x12filenameTe\
    nsorName\x12(\n\x10save_tensor_name\x18\x02\x20\x01(\tR\x0esaveTensorNam\
    e\x12&\n\x0frestore_op_name\x18\x03\x20\x01(\tR\rrestoreOpName\x12\x1e\n\
    \x0bmax_to_keep\x18\x04\x20\x01(\x05R\tmaxToKeep\x12\x18\n\x07sharded\
    \x18\x05\x20\x01(\x08R\x07sharded\x12@\n\x1dkeep_checkpoint_every_n_hour\
    s\x18\x06\x20\x01(\x02R\x19keepCheckpointEveryNHours\x12F\n\x07version\
    \x18\x07\x20\x01(\x0e2,.tensorflow.SaverDef.CheckpointFormatVersionR\x07\
    version\"5\n\x17CheckpointFormatVersion\x12\n\n\x06LEGACY\x10\0\x12\x06\
    \n\x02V1\x10\x01\x12\x06\n\x02V2\x10\x02Be\n\x13org.tensorflow.utilB\x0b\
    SaverProtosP\x01Z<github.com/tensorflow/tensorflow/tensorflow/go/core/pr\
    otobuf\xf8\x01\x01b\x06proto3\
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

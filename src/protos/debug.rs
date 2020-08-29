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
//! Generated file from `tensorflow/core/protobuf/debug.proto`

/// Generated files are compatible only with the same version
/// of protobuf runtime.
// const _PROTOBUF_VERSION_CHECK: () = ::protobuf::VERSION_2_17_0;

#[derive(PartialEq,Clone,Default)]
pub struct DebugTensorWatch {
    // message fields
    pub node_name: ::std::string::String,
    pub output_slot: i32,
    pub debug_ops: ::protobuf::RepeatedField<::std::string::String>,
    pub debug_urls: ::protobuf::RepeatedField<::std::string::String>,
    pub tolerate_debug_op_creation_failures: bool,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a DebugTensorWatch {
    fn default() -> &'a DebugTensorWatch {
        <DebugTensorWatch as ::protobuf::Message>::default_instance()
    }
}

impl DebugTensorWatch {
    pub fn new() -> DebugTensorWatch {
        ::std::default::Default::default()
    }

    // string node_name = 1;


    pub fn get_node_name(&self) -> &str {
        &self.node_name
    }
    pub fn clear_node_name(&mut self) {
        self.node_name.clear();
    }

    // Param is passed by value, moved
    pub fn set_node_name(&mut self, v: ::std::string::String) {
        self.node_name = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_node_name(&mut self) -> &mut ::std::string::String {
        &mut self.node_name
    }

    // Take field
    pub fn take_node_name(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.node_name, ::std::string::String::new())
    }

    // int32 output_slot = 2;


    pub fn get_output_slot(&self) -> i32 {
        self.output_slot
    }
    pub fn clear_output_slot(&mut self) {
        self.output_slot = 0;
    }

    // Param is passed by value, moved
    pub fn set_output_slot(&mut self, v: i32) {
        self.output_slot = v;
    }

    // repeated string debug_ops = 3;


    pub fn get_debug_ops(&self) -> &[::std::string::String] {
        &self.debug_ops
    }
    pub fn clear_debug_ops(&mut self) {
        self.debug_ops.clear();
    }

    // Param is passed by value, moved
    pub fn set_debug_ops(&mut self, v: ::protobuf::RepeatedField<::std::string::String>) {
        self.debug_ops = v;
    }

    // Mutable pointer to the field.
    pub fn mut_debug_ops(&mut self) -> &mut ::protobuf::RepeatedField<::std::string::String> {
        &mut self.debug_ops
    }

    // Take field
    pub fn take_debug_ops(&mut self) -> ::protobuf::RepeatedField<::std::string::String> {
        ::std::mem::replace(&mut self.debug_ops, ::protobuf::RepeatedField::new())
    }

    // repeated string debug_urls = 4;


    pub fn get_debug_urls(&self) -> &[::std::string::String] {
        &self.debug_urls
    }
    pub fn clear_debug_urls(&mut self) {
        self.debug_urls.clear();
    }

    // Param is passed by value, moved
    pub fn set_debug_urls(&mut self, v: ::protobuf::RepeatedField<::std::string::String>) {
        self.debug_urls = v;
    }

    // Mutable pointer to the field.
    pub fn mut_debug_urls(&mut self) -> &mut ::protobuf::RepeatedField<::std::string::String> {
        &mut self.debug_urls
    }

    // Take field
    pub fn take_debug_urls(&mut self) -> ::protobuf::RepeatedField<::std::string::String> {
        ::std::mem::replace(&mut self.debug_urls, ::protobuf::RepeatedField::new())
    }

    // bool tolerate_debug_op_creation_failures = 5;


    pub fn get_tolerate_debug_op_creation_failures(&self) -> bool {
        self.tolerate_debug_op_creation_failures
    }
    pub fn clear_tolerate_debug_op_creation_failures(&mut self) {
        self.tolerate_debug_op_creation_failures = false;
    }

    // Param is passed by value, moved
    pub fn set_tolerate_debug_op_creation_failures(&mut self, v: bool) {
        self.tolerate_debug_op_creation_failures = v;
    }
}

impl ::protobuf::Message for DebugTensorWatch {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.node_name)?;
                },
                2 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int32()?;
                    self.output_slot = tmp;
                },
                3 => {
                    ::protobuf::rt::read_repeated_string_into(wire_type, is, &mut self.debug_ops)?;
                },
                4 => {
                    ::protobuf::rt::read_repeated_string_into(wire_type, is, &mut self.debug_urls)?;
                },
                5 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_bool()?;
                    self.tolerate_debug_op_creation_failures = tmp;
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
        if !self.node_name.is_empty() {
            my_size += ::protobuf::rt::string_size(1, &self.node_name);
        }
        if self.output_slot != 0 {
            my_size += ::protobuf::rt::value_size(2, self.output_slot, ::protobuf::wire_format::WireTypeVarint);
        }
        for value in &self.debug_ops {
            my_size += ::protobuf::rt::string_size(3, &value);
        };
        for value in &self.debug_urls {
            my_size += ::protobuf::rt::string_size(4, &value);
        };
        if self.tolerate_debug_op_creation_failures != false {
            my_size += 2;
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        if !self.node_name.is_empty() {
            os.write_string(1, &self.node_name)?;
        }
        if self.output_slot != 0 {
            os.write_int32(2, self.output_slot)?;
        }
        for v in &self.debug_ops {
            os.write_string(3, &v)?;
        };
        for v in &self.debug_urls {
            os.write_string(4, &v)?;
        };
        if self.tolerate_debug_op_creation_failures != false {
            os.write_bool(5, self.tolerate_debug_op_creation_failures)?;
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

    fn new() -> DebugTensorWatch {
        DebugTensorWatch::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "node_name",
                |m: &DebugTensorWatch| { &m.node_name },
                |m: &mut DebugTensorWatch| { &mut m.node_name },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeInt32>(
                "output_slot",
                |m: &DebugTensorWatch| { &m.output_slot },
                |m: &mut DebugTensorWatch| { &mut m.output_slot },
            ));
            fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "debug_ops",
                |m: &DebugTensorWatch| { &m.debug_ops },
                |m: &mut DebugTensorWatch| { &mut m.debug_ops },
            ));
            fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "debug_urls",
                |m: &DebugTensorWatch| { &m.debug_urls },
                |m: &mut DebugTensorWatch| { &mut m.debug_urls },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeBool>(
                "tolerate_debug_op_creation_failures",
                |m: &DebugTensorWatch| { &m.tolerate_debug_op_creation_failures },
                |m: &mut DebugTensorWatch| { &mut m.tolerate_debug_op_creation_failures },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<DebugTensorWatch>(
                "DebugTensorWatch",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static DebugTensorWatch {
        static instance: ::protobuf::rt::LazyV2<DebugTensorWatch> = ::protobuf::rt::LazyV2::INIT;
        instance.get(DebugTensorWatch::new)
    }
}

impl ::protobuf::Clear for DebugTensorWatch {
    fn clear(&mut self) {
        self.node_name.clear();
        self.output_slot = 0;
        self.debug_ops.clear();
        self.debug_urls.clear();
        self.tolerate_debug_op_creation_failures = false;
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for DebugTensorWatch {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for DebugTensorWatch {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct DebugOptions {
    // message fields
    pub debug_tensor_watch_opts: ::protobuf::RepeatedField<DebugTensorWatch>,
    pub global_step: i64,
    pub reset_disk_byte_usage: bool,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a DebugOptions {
    fn default() -> &'a DebugOptions {
        <DebugOptions as ::protobuf::Message>::default_instance()
    }
}

impl DebugOptions {
    pub fn new() -> DebugOptions {
        ::std::default::Default::default()
    }

    // repeated .tensorflow.DebugTensorWatch debug_tensor_watch_opts = 4;


    pub fn get_debug_tensor_watch_opts(&self) -> &[DebugTensorWatch] {
        &self.debug_tensor_watch_opts
    }
    pub fn clear_debug_tensor_watch_opts(&mut self) {
        self.debug_tensor_watch_opts.clear();
    }

    // Param is passed by value, moved
    pub fn set_debug_tensor_watch_opts(&mut self, v: ::protobuf::RepeatedField<DebugTensorWatch>) {
        self.debug_tensor_watch_opts = v;
    }

    // Mutable pointer to the field.
    pub fn mut_debug_tensor_watch_opts(&mut self) -> &mut ::protobuf::RepeatedField<DebugTensorWatch> {
        &mut self.debug_tensor_watch_opts
    }

    // Take field
    pub fn take_debug_tensor_watch_opts(&mut self) -> ::protobuf::RepeatedField<DebugTensorWatch> {
        ::std::mem::replace(&mut self.debug_tensor_watch_opts, ::protobuf::RepeatedField::new())
    }

    // int64 global_step = 10;


    pub fn get_global_step(&self) -> i64 {
        self.global_step
    }
    pub fn clear_global_step(&mut self) {
        self.global_step = 0;
    }

    // Param is passed by value, moved
    pub fn set_global_step(&mut self, v: i64) {
        self.global_step = v;
    }

    // bool reset_disk_byte_usage = 11;


    pub fn get_reset_disk_byte_usage(&self) -> bool {
        self.reset_disk_byte_usage
    }
    pub fn clear_reset_disk_byte_usage(&mut self) {
        self.reset_disk_byte_usage = false;
    }

    // Param is passed by value, moved
    pub fn set_reset_disk_byte_usage(&mut self, v: bool) {
        self.reset_disk_byte_usage = v;
    }
}

impl ::protobuf::Message for DebugOptions {
    fn is_initialized(&self) -> bool {
        for v in &self.debug_tensor_watch_opts {
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
                4 => {
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.debug_tensor_watch_opts)?;
                },
                10 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int64()?;
                    self.global_step = tmp;
                },
                11 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_bool()?;
                    self.reset_disk_byte_usage = tmp;
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
        for value in &self.debug_tensor_watch_opts {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        if self.global_step != 0 {
            my_size += ::protobuf::rt::value_size(10, self.global_step, ::protobuf::wire_format::WireTypeVarint);
        }
        if self.reset_disk_byte_usage != false {
            my_size += 2;
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        for v in &self.debug_tensor_watch_opts {
            os.write_tag(4, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        };
        if self.global_step != 0 {
            os.write_int64(10, self.global_step)?;
        }
        if self.reset_disk_byte_usage != false {
            os.write_bool(11, self.reset_disk_byte_usage)?;
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

    fn new() -> DebugOptions {
        DebugOptions::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<DebugTensorWatch>>(
                "debug_tensor_watch_opts",
                |m: &DebugOptions| { &m.debug_tensor_watch_opts },
                |m: &mut DebugOptions| { &mut m.debug_tensor_watch_opts },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                "global_step",
                |m: &DebugOptions| { &m.global_step },
                |m: &mut DebugOptions| { &mut m.global_step },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeBool>(
                "reset_disk_byte_usage",
                |m: &DebugOptions| { &m.reset_disk_byte_usage },
                |m: &mut DebugOptions| { &mut m.reset_disk_byte_usage },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<DebugOptions>(
                "DebugOptions",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static DebugOptions {
        static instance: ::protobuf::rt::LazyV2<DebugOptions> = ::protobuf::rt::LazyV2::INIT;
        instance.get(DebugOptions::new)
    }
}

impl ::protobuf::Clear for DebugOptions {
    fn clear(&mut self) {
        self.debug_tensor_watch_opts.clear();
        self.global_step = 0;
        self.reset_disk_byte_usage = false;
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for DebugOptions {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for DebugOptions {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct DebuggedSourceFile {
    // message fields
    pub host: ::std::string::String,
    pub file_path: ::std::string::String,
    pub last_modified: i64,
    pub bytes: i64,
    pub lines: ::protobuf::RepeatedField<::std::string::String>,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a DebuggedSourceFile {
    fn default() -> &'a DebuggedSourceFile {
        <DebuggedSourceFile as ::protobuf::Message>::default_instance()
    }
}

impl DebuggedSourceFile {
    pub fn new() -> DebuggedSourceFile {
        ::std::default::Default::default()
    }

    // string host = 1;


    pub fn get_host(&self) -> &str {
        &self.host
    }
    pub fn clear_host(&mut self) {
        self.host.clear();
    }

    // Param is passed by value, moved
    pub fn set_host(&mut self, v: ::std::string::String) {
        self.host = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_host(&mut self) -> &mut ::std::string::String {
        &mut self.host
    }

    // Take field
    pub fn take_host(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.host, ::std::string::String::new())
    }

    // string file_path = 2;


    pub fn get_file_path(&self) -> &str {
        &self.file_path
    }
    pub fn clear_file_path(&mut self) {
        self.file_path.clear();
    }

    // Param is passed by value, moved
    pub fn set_file_path(&mut self, v: ::std::string::String) {
        self.file_path = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_file_path(&mut self) -> &mut ::std::string::String {
        &mut self.file_path
    }

    // Take field
    pub fn take_file_path(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.file_path, ::std::string::String::new())
    }

    // int64 last_modified = 3;


    pub fn get_last_modified(&self) -> i64 {
        self.last_modified
    }
    pub fn clear_last_modified(&mut self) {
        self.last_modified = 0;
    }

    // Param is passed by value, moved
    pub fn set_last_modified(&mut self, v: i64) {
        self.last_modified = v;
    }

    // int64 bytes = 4;


    pub fn get_bytes(&self) -> i64 {
        self.bytes
    }
    pub fn clear_bytes(&mut self) {
        self.bytes = 0;
    }

    // Param is passed by value, moved
    pub fn set_bytes(&mut self, v: i64) {
        self.bytes = v;
    }

    // repeated string lines = 5;


    pub fn get_lines(&self) -> &[::std::string::String] {
        &self.lines
    }
    pub fn clear_lines(&mut self) {
        self.lines.clear();
    }

    // Param is passed by value, moved
    pub fn set_lines(&mut self, v: ::protobuf::RepeatedField<::std::string::String>) {
        self.lines = v;
    }

    // Mutable pointer to the field.
    pub fn mut_lines(&mut self) -> &mut ::protobuf::RepeatedField<::std::string::String> {
        &mut self.lines
    }

    // Take field
    pub fn take_lines(&mut self) -> ::protobuf::RepeatedField<::std::string::String> {
        ::std::mem::replace(&mut self.lines, ::protobuf::RepeatedField::new())
    }
}

impl ::protobuf::Message for DebuggedSourceFile {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.host)?;
                },
                2 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.file_path)?;
                },
                3 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int64()?;
                    self.last_modified = tmp;
                },
                4 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int64()?;
                    self.bytes = tmp;
                },
                5 => {
                    ::protobuf::rt::read_repeated_string_into(wire_type, is, &mut self.lines)?;
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
        if !self.host.is_empty() {
            my_size += ::protobuf::rt::string_size(1, &self.host);
        }
        if !self.file_path.is_empty() {
            my_size += ::protobuf::rt::string_size(2, &self.file_path);
        }
        if self.last_modified != 0 {
            my_size += ::protobuf::rt::value_size(3, self.last_modified, ::protobuf::wire_format::WireTypeVarint);
        }
        if self.bytes != 0 {
            my_size += ::protobuf::rt::value_size(4, self.bytes, ::protobuf::wire_format::WireTypeVarint);
        }
        for value in &self.lines {
            my_size += ::protobuf::rt::string_size(5, &value);
        };
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        if !self.host.is_empty() {
            os.write_string(1, &self.host)?;
        }
        if !self.file_path.is_empty() {
            os.write_string(2, &self.file_path)?;
        }
        if self.last_modified != 0 {
            os.write_int64(3, self.last_modified)?;
        }
        if self.bytes != 0 {
            os.write_int64(4, self.bytes)?;
        }
        for v in &self.lines {
            os.write_string(5, &v)?;
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

    fn new() -> DebuggedSourceFile {
        DebuggedSourceFile::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "host",
                |m: &DebuggedSourceFile| { &m.host },
                |m: &mut DebuggedSourceFile| { &mut m.host },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "file_path",
                |m: &DebuggedSourceFile| { &m.file_path },
                |m: &mut DebuggedSourceFile| { &mut m.file_path },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                "last_modified",
                |m: &DebuggedSourceFile| { &m.last_modified },
                |m: &mut DebuggedSourceFile| { &mut m.last_modified },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                "bytes",
                |m: &DebuggedSourceFile| { &m.bytes },
                |m: &mut DebuggedSourceFile| { &mut m.bytes },
            ));
            fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "lines",
                |m: &DebuggedSourceFile| { &m.lines },
                |m: &mut DebuggedSourceFile| { &mut m.lines },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<DebuggedSourceFile>(
                "DebuggedSourceFile",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static DebuggedSourceFile {
        static instance: ::protobuf::rt::LazyV2<DebuggedSourceFile> = ::protobuf::rt::LazyV2::INIT;
        instance.get(DebuggedSourceFile::new)
    }
}

impl ::protobuf::Clear for DebuggedSourceFile {
    fn clear(&mut self) {
        self.host.clear();
        self.file_path.clear();
        self.last_modified = 0;
        self.bytes = 0;
        self.lines.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for DebuggedSourceFile {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for DebuggedSourceFile {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct DebuggedSourceFiles {
    // message fields
    pub source_files: ::protobuf::RepeatedField<DebuggedSourceFile>,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a DebuggedSourceFiles {
    fn default() -> &'a DebuggedSourceFiles {
        <DebuggedSourceFiles as ::protobuf::Message>::default_instance()
    }
}

impl DebuggedSourceFiles {
    pub fn new() -> DebuggedSourceFiles {
        ::std::default::Default::default()
    }

    // repeated .tensorflow.DebuggedSourceFile source_files = 1;


    pub fn get_source_files(&self) -> &[DebuggedSourceFile] {
        &self.source_files
    }
    pub fn clear_source_files(&mut self) {
        self.source_files.clear();
    }

    // Param is passed by value, moved
    pub fn set_source_files(&mut self, v: ::protobuf::RepeatedField<DebuggedSourceFile>) {
        self.source_files = v;
    }

    // Mutable pointer to the field.
    pub fn mut_source_files(&mut self) -> &mut ::protobuf::RepeatedField<DebuggedSourceFile> {
        &mut self.source_files
    }

    // Take field
    pub fn take_source_files(&mut self) -> ::protobuf::RepeatedField<DebuggedSourceFile> {
        ::std::mem::replace(&mut self.source_files, ::protobuf::RepeatedField::new())
    }
}

impl ::protobuf::Message for DebuggedSourceFiles {
    fn is_initialized(&self) -> bool {
        for v in &self.source_files {
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
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.source_files)?;
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
        for value in &self.source_files {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        for v in &self.source_files {
            os.write_tag(1, ::protobuf::wire_format::WireTypeLengthDelimited)?;
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

    fn new() -> DebuggedSourceFiles {
        DebuggedSourceFiles::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<DebuggedSourceFile>>(
                "source_files",
                |m: &DebuggedSourceFiles| { &m.source_files },
                |m: &mut DebuggedSourceFiles| { &mut m.source_files },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<DebuggedSourceFiles>(
                "DebuggedSourceFiles",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static DebuggedSourceFiles {
        static instance: ::protobuf::rt::LazyV2<DebuggedSourceFiles> = ::protobuf::rt::LazyV2::INIT;
        instance.get(DebuggedSourceFiles::new)
    }
}

impl ::protobuf::Clear for DebuggedSourceFiles {
    fn clear(&mut self) {
        self.source_files.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for DebuggedSourceFiles {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for DebuggedSourceFiles {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

static file_descriptor_proto_data: &'static [u8] = b"\
    \n$tensorflow/core/protobuf/debug.proto\x12\ntensorflow\"\xda\x01\n\x10D\
    ebugTensorWatch\x12\x1b\n\tnode_name\x18\x01\x20\x01(\tR\x08nodeName\x12\
    \x1f\n\x0boutput_slot\x18\x02\x20\x01(\x05R\noutputSlot\x12\x1b\n\tdebug\
    _ops\x18\x03\x20\x03(\tR\x08debugOps\x12\x1d\n\ndebug_urls\x18\x04\x20\
    \x03(\tR\tdebugUrls\x12L\n#tolerate_debug_op_creation_failures\x18\x05\
    \x20\x01(\x08R\x1ftolerateDebugOpCreationFailures\"\xb7\x01\n\x0cDebugOp\
    tions\x12S\n\x17debug_tensor_watch_opts\x18\x04\x20\x03(\x0b2\x1c.tensor\
    flow.DebugTensorWatchR\x14debugTensorWatchOpts\x12\x1f\n\x0bglobal_step\
    \x18\n\x20\x01(\x03R\nglobalStep\x121\n\x15reset_disk_byte_usage\x18\x0b\
    \x20\x01(\x08R\x12resetDiskByteUsage\"\x96\x01\n\x12DebuggedSourceFile\
    \x12\x12\n\x04host\x18\x01\x20\x01(\tR\x04host\x12\x1b\n\tfile_path\x18\
    \x02\x20\x01(\tR\x08filePath\x12#\n\rlast_modified\x18\x03\x20\x01(\x03R\
    \x0clastModified\x12\x14\n\x05bytes\x18\x04\x20\x01(\x03R\x05bytes\x12\
    \x14\n\x05lines\x18\x05\x20\x03(\tR\x05lines\"X\n\x13DebuggedSourceFiles\
    \x12A\n\x0csource_files\x18\x01\x20\x03(\x0b2\x1e.tensorflow.DebuggedSou\
    rceFileR\x0bsourceFilesBj\n\x18org.tensorflow.frameworkB\x0bDebugProtosP\
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

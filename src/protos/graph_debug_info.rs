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
//! Generated file from `tensorflow/core/protobuf/graph_debug_info.proto`

/// Generated files are compatible only with the same version
/// of protobuf runtime.
// const _PROTOBUF_VERSION_CHECK: () = ::protobuf::VERSION_2_17_0;

#[derive(PartialEq,Clone,Default)]
pub struct GraphDebugInfo {
    // message fields
    pub files: ::protobuf::RepeatedField<::std::string::String>,
    pub traces: ::std::collections::HashMap<::std::string::String, GraphDebugInfo_StackTrace>,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a GraphDebugInfo {
    fn default() -> &'a GraphDebugInfo {
        <GraphDebugInfo as ::protobuf::Message>::default_instance()
    }
}

impl GraphDebugInfo {
    pub fn new() -> GraphDebugInfo {
        ::std::default::Default::default()
    }

    // repeated string files = 1;


    pub fn get_files(&self) -> &[::std::string::String] {
        &self.files
    }
    pub fn clear_files(&mut self) {
        self.files.clear();
    }

    // Param is passed by value, moved
    pub fn set_files(&mut self, v: ::protobuf::RepeatedField<::std::string::String>) {
        self.files = v;
    }

    // Mutable pointer to the field.
    pub fn mut_files(&mut self) -> &mut ::protobuf::RepeatedField<::std::string::String> {
        &mut self.files
    }

    // Take field
    pub fn take_files(&mut self) -> ::protobuf::RepeatedField<::std::string::String> {
        ::std::mem::replace(&mut self.files, ::protobuf::RepeatedField::new())
    }

    // repeated .tensorflow.GraphDebugInfo.TracesEntry traces = 2;


    pub fn get_traces(&self) -> &::std::collections::HashMap<::std::string::String, GraphDebugInfo_StackTrace> {
        &self.traces
    }
    pub fn clear_traces(&mut self) {
        self.traces.clear();
    }

    // Param is passed by value, moved
    pub fn set_traces(&mut self, v: ::std::collections::HashMap<::std::string::String, GraphDebugInfo_StackTrace>) {
        self.traces = v;
    }

    // Mutable pointer to the field.
    pub fn mut_traces(&mut self) -> &mut ::std::collections::HashMap<::std::string::String, GraphDebugInfo_StackTrace> {
        &mut self.traces
    }

    // Take field
    pub fn take_traces(&mut self) -> ::std::collections::HashMap<::std::string::String, GraphDebugInfo_StackTrace> {
        ::std::mem::replace(&mut self.traces, ::std::collections::HashMap::new())
    }
}

impl ::protobuf::Message for GraphDebugInfo {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_repeated_string_into(wire_type, is, &mut self.files)?;
                },
                2 => {
                    ::protobuf::rt::read_map_into::<::protobuf::types::ProtobufTypeString, ::protobuf::types::ProtobufTypeMessage<GraphDebugInfo_StackTrace>>(wire_type, is, &mut self.traces)?;
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
        for value in &self.files {
            my_size += ::protobuf::rt::string_size(1, &value);
        };
        my_size += ::protobuf::rt::compute_map_size::<::protobuf::types::ProtobufTypeString, ::protobuf::types::ProtobufTypeMessage<GraphDebugInfo_StackTrace>>(2, &self.traces);
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        for v in &self.files {
            os.write_string(1, &v)?;
        };
        ::protobuf::rt::write_map_with_cached_sizes::<::protobuf::types::ProtobufTypeString, ::protobuf::types::ProtobufTypeMessage<GraphDebugInfo_StackTrace>>(2, &self.traces, os)?;
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

    fn new() -> GraphDebugInfo {
        GraphDebugInfo::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "files",
                |m: &GraphDebugInfo| { &m.files },
                |m: &mut GraphDebugInfo| { &mut m.files },
            ));
            fields.push(::protobuf::reflect::accessor::make_map_accessor::<_, ::protobuf::types::ProtobufTypeString, ::protobuf::types::ProtobufTypeMessage<GraphDebugInfo_StackTrace>>(
                "traces",
                |m: &GraphDebugInfo| { &m.traces },
                |m: &mut GraphDebugInfo| { &mut m.traces },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<GraphDebugInfo>(
                "GraphDebugInfo",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static GraphDebugInfo {
        static instance: ::protobuf::rt::LazyV2<GraphDebugInfo> = ::protobuf::rt::LazyV2::INIT;
        instance.get(GraphDebugInfo::new)
    }
}

impl ::protobuf::Clear for GraphDebugInfo {
    fn clear(&mut self) {
        self.files.clear();
        self.traces.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for GraphDebugInfo {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for GraphDebugInfo {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct GraphDebugInfo_FileLineCol {
    // message fields
    pub file_index: i32,
    pub line: i32,
    pub col: i32,
    pub func: ::std::string::String,
    pub code: ::std::string::String,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a GraphDebugInfo_FileLineCol {
    fn default() -> &'a GraphDebugInfo_FileLineCol {
        <GraphDebugInfo_FileLineCol as ::protobuf::Message>::default_instance()
    }
}

impl GraphDebugInfo_FileLineCol {
    pub fn new() -> GraphDebugInfo_FileLineCol {
        ::std::default::Default::default()
    }

    // int32 file_index = 1;


    pub fn get_file_index(&self) -> i32 {
        self.file_index
    }
    pub fn clear_file_index(&mut self) {
        self.file_index = 0;
    }

    // Param is passed by value, moved
    pub fn set_file_index(&mut self, v: i32) {
        self.file_index = v;
    }

    // int32 line = 2;


    pub fn get_line(&self) -> i32 {
        self.line
    }
    pub fn clear_line(&mut self) {
        self.line = 0;
    }

    // Param is passed by value, moved
    pub fn set_line(&mut self, v: i32) {
        self.line = v;
    }

    // int32 col = 3;


    pub fn get_col(&self) -> i32 {
        self.col
    }
    pub fn clear_col(&mut self) {
        self.col = 0;
    }

    // Param is passed by value, moved
    pub fn set_col(&mut self, v: i32) {
        self.col = v;
    }

    // string func = 4;


    pub fn get_func(&self) -> &str {
        &self.func
    }
    pub fn clear_func(&mut self) {
        self.func.clear();
    }

    // Param is passed by value, moved
    pub fn set_func(&mut self, v: ::std::string::String) {
        self.func = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_func(&mut self) -> &mut ::std::string::String {
        &mut self.func
    }

    // Take field
    pub fn take_func(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.func, ::std::string::String::new())
    }

    // string code = 5;


    pub fn get_code(&self) -> &str {
        &self.code
    }
    pub fn clear_code(&mut self) {
        self.code.clear();
    }

    // Param is passed by value, moved
    pub fn set_code(&mut self, v: ::std::string::String) {
        self.code = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_code(&mut self) -> &mut ::std::string::String {
        &mut self.code
    }

    // Take field
    pub fn take_code(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.code, ::std::string::String::new())
    }
}

impl ::protobuf::Message for GraphDebugInfo_FileLineCol {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int32()?;
                    self.file_index = tmp;
                },
                2 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int32()?;
                    self.line = tmp;
                },
                3 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int32()?;
                    self.col = tmp;
                },
                4 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.func)?;
                },
                5 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.code)?;
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
        if self.file_index != 0 {
            my_size += ::protobuf::rt::value_size(1, self.file_index, ::protobuf::wire_format::WireTypeVarint);
        }
        if self.line != 0 {
            my_size += ::protobuf::rt::value_size(2, self.line, ::protobuf::wire_format::WireTypeVarint);
        }
        if self.col != 0 {
            my_size += ::protobuf::rt::value_size(3, self.col, ::protobuf::wire_format::WireTypeVarint);
        }
        if !self.func.is_empty() {
            my_size += ::protobuf::rt::string_size(4, &self.func);
        }
        if !self.code.is_empty() {
            my_size += ::protobuf::rt::string_size(5, &self.code);
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        if self.file_index != 0 {
            os.write_int32(1, self.file_index)?;
        }
        if self.line != 0 {
            os.write_int32(2, self.line)?;
        }
        if self.col != 0 {
            os.write_int32(3, self.col)?;
        }
        if !self.func.is_empty() {
            os.write_string(4, &self.func)?;
        }
        if !self.code.is_empty() {
            os.write_string(5, &self.code)?;
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

    fn new() -> GraphDebugInfo_FileLineCol {
        GraphDebugInfo_FileLineCol::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeInt32>(
                "file_index",
                |m: &GraphDebugInfo_FileLineCol| { &m.file_index },
                |m: &mut GraphDebugInfo_FileLineCol| { &mut m.file_index },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeInt32>(
                "line",
                |m: &GraphDebugInfo_FileLineCol| { &m.line },
                |m: &mut GraphDebugInfo_FileLineCol| { &mut m.line },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeInt32>(
                "col",
                |m: &GraphDebugInfo_FileLineCol| { &m.col },
                |m: &mut GraphDebugInfo_FileLineCol| { &mut m.col },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "func",
                |m: &GraphDebugInfo_FileLineCol| { &m.func },
                |m: &mut GraphDebugInfo_FileLineCol| { &mut m.func },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "code",
                |m: &GraphDebugInfo_FileLineCol| { &m.code },
                |m: &mut GraphDebugInfo_FileLineCol| { &mut m.code },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<GraphDebugInfo_FileLineCol>(
                "GraphDebugInfo.FileLineCol",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static GraphDebugInfo_FileLineCol {
        static instance: ::protobuf::rt::LazyV2<GraphDebugInfo_FileLineCol> = ::protobuf::rt::LazyV2::INIT;
        instance.get(GraphDebugInfo_FileLineCol::new)
    }
}

impl ::protobuf::Clear for GraphDebugInfo_FileLineCol {
    fn clear(&mut self) {
        self.file_index = 0;
        self.line = 0;
        self.col = 0;
        self.func.clear();
        self.code.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for GraphDebugInfo_FileLineCol {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for GraphDebugInfo_FileLineCol {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct GraphDebugInfo_StackTrace {
    // message fields
    pub file_line_cols: ::protobuf::RepeatedField<GraphDebugInfo_FileLineCol>,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a GraphDebugInfo_StackTrace {
    fn default() -> &'a GraphDebugInfo_StackTrace {
        <GraphDebugInfo_StackTrace as ::protobuf::Message>::default_instance()
    }
}

impl GraphDebugInfo_StackTrace {
    pub fn new() -> GraphDebugInfo_StackTrace {
        ::std::default::Default::default()
    }

    // repeated .tensorflow.GraphDebugInfo.FileLineCol file_line_cols = 1;


    pub fn get_file_line_cols(&self) -> &[GraphDebugInfo_FileLineCol] {
        &self.file_line_cols
    }
    pub fn clear_file_line_cols(&mut self) {
        self.file_line_cols.clear();
    }

    // Param is passed by value, moved
    pub fn set_file_line_cols(&mut self, v: ::protobuf::RepeatedField<GraphDebugInfo_FileLineCol>) {
        self.file_line_cols = v;
    }

    // Mutable pointer to the field.
    pub fn mut_file_line_cols(&mut self) -> &mut ::protobuf::RepeatedField<GraphDebugInfo_FileLineCol> {
        &mut self.file_line_cols
    }

    // Take field
    pub fn take_file_line_cols(&mut self) -> ::protobuf::RepeatedField<GraphDebugInfo_FileLineCol> {
        ::std::mem::replace(&mut self.file_line_cols, ::protobuf::RepeatedField::new())
    }
}

impl ::protobuf::Message for GraphDebugInfo_StackTrace {
    fn is_initialized(&self) -> bool {
        for v in &self.file_line_cols {
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
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.file_line_cols)?;
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
        for value in &self.file_line_cols {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        for v in &self.file_line_cols {
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

    fn new() -> GraphDebugInfo_StackTrace {
        GraphDebugInfo_StackTrace::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<GraphDebugInfo_FileLineCol>>(
                "file_line_cols",
                |m: &GraphDebugInfo_StackTrace| { &m.file_line_cols },
                |m: &mut GraphDebugInfo_StackTrace| { &mut m.file_line_cols },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<GraphDebugInfo_StackTrace>(
                "GraphDebugInfo.StackTrace",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static GraphDebugInfo_StackTrace {
        static instance: ::protobuf::rt::LazyV2<GraphDebugInfo_StackTrace> = ::protobuf::rt::LazyV2::INIT;
        instance.get(GraphDebugInfo_StackTrace::new)
    }
}

impl ::protobuf::Clear for GraphDebugInfo_StackTrace {
    fn clear(&mut self) {
        self.file_line_cols.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for GraphDebugInfo_StackTrace {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for GraphDebugInfo_StackTrace {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

static file_descriptor_proto_data: &'static [u8] = b"\
    \n/tensorflow/core/protobuf/graph_debug_info.proto\x12\ntensorflow\"\xa0\
    \x03\n\x0eGraphDebugInfo\x12\x14\n\x05files\x18\x01\x20\x03(\tR\x05files\
    \x12>\n\x06traces\x18\x02\x20\x03(\x0b2&.tensorflow.GraphDebugInfo.Trace\
    sEntryR\x06traces\x1az\n\x0bFileLineCol\x12\x1d\n\nfile_index\x18\x01\
    \x20\x01(\x05R\tfileIndex\x12\x12\n\x04line\x18\x02\x20\x01(\x05R\x04lin\
    e\x12\x10\n\x03col\x18\x03\x20\x01(\x05R\x03col\x12\x12\n\x04func\x18\
    \x04\x20\x01(\tR\x04func\x12\x12\n\x04code\x18\x05\x20\x01(\tR\x04code\
    \x1aZ\n\nStackTrace\x12L\n\x0efile_line_cols\x18\x01\x20\x03(\x0b2&.tens\
    orflow.GraphDebugInfo.FileLineColR\x0cfileLineCols\x1a`\n\x0bTracesEntry\
    \x12\x10\n\x03key\x18\x01\x20\x01(\tR\x03key\x12;\n\x05value\x18\x02\x20\
    \x01(\x0b2%.tensorflow.GraphDebugInfo.StackTraceR\x05value:\x028\x01B5\n\
    \x18org.tensorflow.frameworkB\x14GraphDebugInfoProtosP\x01\xf8\x01\x01b\
    \x06proto3\
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

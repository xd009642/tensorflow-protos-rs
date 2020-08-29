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
//! Generated file from `tensorflow/core/framework/variable.proto`

/// Generated files are compatible only with the same version
/// of protobuf runtime.
// const _PROTOBUF_VERSION_CHECK: () = ::protobuf::VERSION_2_17_0;

#[derive(PartialEq,Clone,Default)]
pub struct VariableDef {
    // message fields
    pub variable_name: ::std::string::String,
    pub initial_value_name: ::std::string::String,
    pub initializer_name: ::std::string::String,
    pub snapshot_name: ::std::string::String,
    pub save_slice_info_def: ::protobuf::SingularPtrField<SaveSliceInfoDef>,
    pub is_resource: bool,
    pub trainable: bool,
    pub synchronization: VariableSynchronization,
    pub aggregation: VariableAggregation,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a VariableDef {
    fn default() -> &'a VariableDef {
        <VariableDef as ::protobuf::Message>::default_instance()
    }
}

impl VariableDef {
    pub fn new() -> VariableDef {
        ::std::default::Default::default()
    }

    // string variable_name = 1;


    pub fn get_variable_name(&self) -> &str {
        &self.variable_name
    }
    pub fn clear_variable_name(&mut self) {
        self.variable_name.clear();
    }

    // Param is passed by value, moved
    pub fn set_variable_name(&mut self, v: ::std::string::String) {
        self.variable_name = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_variable_name(&mut self) -> &mut ::std::string::String {
        &mut self.variable_name
    }

    // Take field
    pub fn take_variable_name(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.variable_name, ::std::string::String::new())
    }

    // string initial_value_name = 6;


    pub fn get_initial_value_name(&self) -> &str {
        &self.initial_value_name
    }
    pub fn clear_initial_value_name(&mut self) {
        self.initial_value_name.clear();
    }

    // Param is passed by value, moved
    pub fn set_initial_value_name(&mut self, v: ::std::string::String) {
        self.initial_value_name = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_initial_value_name(&mut self) -> &mut ::std::string::String {
        &mut self.initial_value_name
    }

    // Take field
    pub fn take_initial_value_name(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.initial_value_name, ::std::string::String::new())
    }

    // string initializer_name = 2;


    pub fn get_initializer_name(&self) -> &str {
        &self.initializer_name
    }
    pub fn clear_initializer_name(&mut self) {
        self.initializer_name.clear();
    }

    // Param is passed by value, moved
    pub fn set_initializer_name(&mut self, v: ::std::string::String) {
        self.initializer_name = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_initializer_name(&mut self) -> &mut ::std::string::String {
        &mut self.initializer_name
    }

    // Take field
    pub fn take_initializer_name(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.initializer_name, ::std::string::String::new())
    }

    // string snapshot_name = 3;


    pub fn get_snapshot_name(&self) -> &str {
        &self.snapshot_name
    }
    pub fn clear_snapshot_name(&mut self) {
        self.snapshot_name.clear();
    }

    // Param is passed by value, moved
    pub fn set_snapshot_name(&mut self, v: ::std::string::String) {
        self.snapshot_name = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_snapshot_name(&mut self) -> &mut ::std::string::String {
        &mut self.snapshot_name
    }

    // Take field
    pub fn take_snapshot_name(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.snapshot_name, ::std::string::String::new())
    }

    // .tensorflow.SaveSliceInfoDef save_slice_info_def = 4;


    pub fn get_save_slice_info_def(&self) -> &SaveSliceInfoDef {
        self.save_slice_info_def.as_ref().unwrap_or_else(|| <SaveSliceInfoDef as ::protobuf::Message>::default_instance())
    }
    pub fn clear_save_slice_info_def(&mut self) {
        self.save_slice_info_def.clear();
    }

    pub fn has_save_slice_info_def(&self) -> bool {
        self.save_slice_info_def.is_some()
    }

    // Param is passed by value, moved
    pub fn set_save_slice_info_def(&mut self, v: SaveSliceInfoDef) {
        self.save_slice_info_def = ::protobuf::SingularPtrField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_save_slice_info_def(&mut self) -> &mut SaveSliceInfoDef {
        if self.save_slice_info_def.is_none() {
            self.save_slice_info_def.set_default();
        }
        self.save_slice_info_def.as_mut().unwrap()
    }

    // Take field
    pub fn take_save_slice_info_def(&mut self) -> SaveSliceInfoDef {
        self.save_slice_info_def.take().unwrap_or_else(|| SaveSliceInfoDef::new())
    }

    // bool is_resource = 5;


    pub fn get_is_resource(&self) -> bool {
        self.is_resource
    }
    pub fn clear_is_resource(&mut self) {
        self.is_resource = false;
    }

    // Param is passed by value, moved
    pub fn set_is_resource(&mut self, v: bool) {
        self.is_resource = v;
    }

    // bool trainable = 7;


    pub fn get_trainable(&self) -> bool {
        self.trainable
    }
    pub fn clear_trainable(&mut self) {
        self.trainable = false;
    }

    // Param is passed by value, moved
    pub fn set_trainable(&mut self, v: bool) {
        self.trainable = v;
    }

    // .tensorflow.VariableSynchronization synchronization = 8;


    pub fn get_synchronization(&self) -> VariableSynchronization {
        self.synchronization
    }
    pub fn clear_synchronization(&mut self) {
        self.synchronization = VariableSynchronization::VARIABLE_SYNCHRONIZATION_AUTO;
    }

    // Param is passed by value, moved
    pub fn set_synchronization(&mut self, v: VariableSynchronization) {
        self.synchronization = v;
    }

    // .tensorflow.VariableAggregation aggregation = 9;


    pub fn get_aggregation(&self) -> VariableAggregation {
        self.aggregation
    }
    pub fn clear_aggregation(&mut self) {
        self.aggregation = VariableAggregation::VARIABLE_AGGREGATION_NONE;
    }

    // Param is passed by value, moved
    pub fn set_aggregation(&mut self, v: VariableAggregation) {
        self.aggregation = v;
    }
}

impl ::protobuf::Message for VariableDef {
    fn is_initialized(&self) -> bool {
        for v in &self.save_slice_info_def {
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
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.variable_name)?;
                },
                6 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.initial_value_name)?;
                },
                2 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.initializer_name)?;
                },
                3 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.snapshot_name)?;
                },
                4 => {
                    ::protobuf::rt::read_singular_message_into(wire_type, is, &mut self.save_slice_info_def)?;
                },
                5 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_bool()?;
                    self.is_resource = tmp;
                },
                7 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_bool()?;
                    self.trainable = tmp;
                },
                8 => {
                    ::protobuf::rt::read_proto3_enum_with_unknown_fields_into(wire_type, is, &mut self.synchronization, 8, &mut self.unknown_fields)?
                },
                9 => {
                    ::protobuf::rt::read_proto3_enum_with_unknown_fields_into(wire_type, is, &mut self.aggregation, 9, &mut self.unknown_fields)?
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
        if !self.variable_name.is_empty() {
            my_size += ::protobuf::rt::string_size(1, &self.variable_name);
        }
        if !self.initial_value_name.is_empty() {
            my_size += ::protobuf::rt::string_size(6, &self.initial_value_name);
        }
        if !self.initializer_name.is_empty() {
            my_size += ::protobuf::rt::string_size(2, &self.initializer_name);
        }
        if !self.snapshot_name.is_empty() {
            my_size += ::protobuf::rt::string_size(3, &self.snapshot_name);
        }
        if let Some(ref v) = self.save_slice_info_def.as_ref() {
            let len = v.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        }
        if self.is_resource != false {
            my_size += 2;
        }
        if self.trainable != false {
            my_size += 2;
        }
        if self.synchronization != VariableSynchronization::VARIABLE_SYNCHRONIZATION_AUTO {
            my_size += ::protobuf::rt::enum_size(8, self.synchronization);
        }
        if self.aggregation != VariableAggregation::VARIABLE_AGGREGATION_NONE {
            my_size += ::protobuf::rt::enum_size(9, self.aggregation);
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        if !self.variable_name.is_empty() {
            os.write_string(1, &self.variable_name)?;
        }
        if !self.initial_value_name.is_empty() {
            os.write_string(6, &self.initial_value_name)?;
        }
        if !self.initializer_name.is_empty() {
            os.write_string(2, &self.initializer_name)?;
        }
        if !self.snapshot_name.is_empty() {
            os.write_string(3, &self.snapshot_name)?;
        }
        if let Some(ref v) = self.save_slice_info_def.as_ref() {
            os.write_tag(4, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        }
        if self.is_resource != false {
            os.write_bool(5, self.is_resource)?;
        }
        if self.trainable != false {
            os.write_bool(7, self.trainable)?;
        }
        if self.synchronization != VariableSynchronization::VARIABLE_SYNCHRONIZATION_AUTO {
            os.write_enum(8, ::protobuf::ProtobufEnum::value(&self.synchronization))?;
        }
        if self.aggregation != VariableAggregation::VARIABLE_AGGREGATION_NONE {
            os.write_enum(9, ::protobuf::ProtobufEnum::value(&self.aggregation))?;
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

    fn new() -> VariableDef {
        VariableDef::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "variable_name",
                |m: &VariableDef| { &m.variable_name },
                |m: &mut VariableDef| { &mut m.variable_name },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "initial_value_name",
                |m: &VariableDef| { &m.initial_value_name },
                |m: &mut VariableDef| { &mut m.initial_value_name },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "initializer_name",
                |m: &VariableDef| { &m.initializer_name },
                |m: &mut VariableDef| { &mut m.initializer_name },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "snapshot_name",
                |m: &VariableDef| { &m.snapshot_name },
                |m: &mut VariableDef| { &mut m.snapshot_name },
            ));
            fields.push(::protobuf::reflect::accessor::make_singular_ptr_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<SaveSliceInfoDef>>(
                "save_slice_info_def",
                |m: &VariableDef| { &m.save_slice_info_def },
                |m: &mut VariableDef| { &mut m.save_slice_info_def },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeBool>(
                "is_resource",
                |m: &VariableDef| { &m.is_resource },
                |m: &mut VariableDef| { &mut m.is_resource },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeBool>(
                "trainable",
                |m: &VariableDef| { &m.trainable },
                |m: &mut VariableDef| { &mut m.trainable },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeEnum<VariableSynchronization>>(
                "synchronization",
                |m: &VariableDef| { &m.synchronization },
                |m: &mut VariableDef| { &mut m.synchronization },
            ));
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeEnum<VariableAggregation>>(
                "aggregation",
                |m: &VariableDef| { &m.aggregation },
                |m: &mut VariableDef| { &mut m.aggregation },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<VariableDef>(
                "VariableDef",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static VariableDef {
        static instance: ::protobuf::rt::LazyV2<VariableDef> = ::protobuf::rt::LazyV2::INIT;
        instance.get(VariableDef::new)
    }
}

impl ::protobuf::Clear for VariableDef {
    fn clear(&mut self) {
        self.variable_name.clear();
        self.initial_value_name.clear();
        self.initializer_name.clear();
        self.snapshot_name.clear();
        self.save_slice_info_def.clear();
        self.is_resource = false;
        self.trainable = false;
        self.synchronization = VariableSynchronization::VARIABLE_SYNCHRONIZATION_AUTO;
        self.aggregation = VariableAggregation::VARIABLE_AGGREGATION_NONE;
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for VariableDef {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for VariableDef {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct SaveSliceInfoDef {
    // message fields
    pub full_name: ::std::string::String,
    pub full_shape: ::std::vec::Vec<i64>,
    pub var_offset: ::std::vec::Vec<i64>,
    pub var_shape: ::std::vec::Vec<i64>,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a SaveSliceInfoDef {
    fn default() -> &'a SaveSliceInfoDef {
        <SaveSliceInfoDef as ::protobuf::Message>::default_instance()
    }
}

impl SaveSliceInfoDef {
    pub fn new() -> SaveSliceInfoDef {
        ::std::default::Default::default()
    }

    // string full_name = 1;


    pub fn get_full_name(&self) -> &str {
        &self.full_name
    }
    pub fn clear_full_name(&mut self) {
        self.full_name.clear();
    }

    // Param is passed by value, moved
    pub fn set_full_name(&mut self, v: ::std::string::String) {
        self.full_name = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_full_name(&mut self) -> &mut ::std::string::String {
        &mut self.full_name
    }

    // Take field
    pub fn take_full_name(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.full_name, ::std::string::String::new())
    }

    // repeated int64 full_shape = 2;


    pub fn get_full_shape(&self) -> &[i64] {
        &self.full_shape
    }
    pub fn clear_full_shape(&mut self) {
        self.full_shape.clear();
    }

    // Param is passed by value, moved
    pub fn set_full_shape(&mut self, v: ::std::vec::Vec<i64>) {
        self.full_shape = v;
    }

    // Mutable pointer to the field.
    pub fn mut_full_shape(&mut self) -> &mut ::std::vec::Vec<i64> {
        &mut self.full_shape
    }

    // Take field
    pub fn take_full_shape(&mut self) -> ::std::vec::Vec<i64> {
        ::std::mem::replace(&mut self.full_shape, ::std::vec::Vec::new())
    }

    // repeated int64 var_offset = 3;


    pub fn get_var_offset(&self) -> &[i64] {
        &self.var_offset
    }
    pub fn clear_var_offset(&mut self) {
        self.var_offset.clear();
    }

    // Param is passed by value, moved
    pub fn set_var_offset(&mut self, v: ::std::vec::Vec<i64>) {
        self.var_offset = v;
    }

    // Mutable pointer to the field.
    pub fn mut_var_offset(&mut self) -> &mut ::std::vec::Vec<i64> {
        &mut self.var_offset
    }

    // Take field
    pub fn take_var_offset(&mut self) -> ::std::vec::Vec<i64> {
        ::std::mem::replace(&mut self.var_offset, ::std::vec::Vec::new())
    }

    // repeated int64 var_shape = 4;


    pub fn get_var_shape(&self) -> &[i64] {
        &self.var_shape
    }
    pub fn clear_var_shape(&mut self) {
        self.var_shape.clear();
    }

    // Param is passed by value, moved
    pub fn set_var_shape(&mut self, v: ::std::vec::Vec<i64>) {
        self.var_shape = v;
    }

    // Mutable pointer to the field.
    pub fn mut_var_shape(&mut self) -> &mut ::std::vec::Vec<i64> {
        &mut self.var_shape
    }

    // Take field
    pub fn take_var_shape(&mut self) -> ::std::vec::Vec<i64> {
        ::std::mem::replace(&mut self.var_shape, ::std::vec::Vec::new())
    }
}

impl ::protobuf::Message for SaveSliceInfoDef {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.full_name)?;
                },
                2 => {
                    ::protobuf::rt::read_repeated_int64_into(wire_type, is, &mut self.full_shape)?;
                },
                3 => {
                    ::protobuf::rt::read_repeated_int64_into(wire_type, is, &mut self.var_offset)?;
                },
                4 => {
                    ::protobuf::rt::read_repeated_int64_into(wire_type, is, &mut self.var_shape)?;
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
        if !self.full_name.is_empty() {
            my_size += ::protobuf::rt::string_size(1, &self.full_name);
        }
        for value in &self.full_shape {
            my_size += ::protobuf::rt::value_size(2, *value, ::protobuf::wire_format::WireTypeVarint);
        };
        for value in &self.var_offset {
            my_size += ::protobuf::rt::value_size(3, *value, ::protobuf::wire_format::WireTypeVarint);
        };
        for value in &self.var_shape {
            my_size += ::protobuf::rt::value_size(4, *value, ::protobuf::wire_format::WireTypeVarint);
        };
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        if !self.full_name.is_empty() {
            os.write_string(1, &self.full_name)?;
        }
        for v in &self.full_shape {
            os.write_int64(2, *v)?;
        };
        for v in &self.var_offset {
            os.write_int64(3, *v)?;
        };
        for v in &self.var_shape {
            os.write_int64(4, *v)?;
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

    fn new() -> SaveSliceInfoDef {
        SaveSliceInfoDef::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                "full_name",
                |m: &SaveSliceInfoDef| { &m.full_name },
                |m: &mut SaveSliceInfoDef| { &mut m.full_name },
            ));
            fields.push(::protobuf::reflect::accessor::make_vec_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                "full_shape",
                |m: &SaveSliceInfoDef| { &m.full_shape },
                |m: &mut SaveSliceInfoDef| { &mut m.full_shape },
            ));
            fields.push(::protobuf::reflect::accessor::make_vec_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                "var_offset",
                |m: &SaveSliceInfoDef| { &m.var_offset },
                |m: &mut SaveSliceInfoDef| { &mut m.var_offset },
            ));
            fields.push(::protobuf::reflect::accessor::make_vec_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                "var_shape",
                |m: &SaveSliceInfoDef| { &m.var_shape },
                |m: &mut SaveSliceInfoDef| { &mut m.var_shape },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<SaveSliceInfoDef>(
                "SaveSliceInfoDef",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static SaveSliceInfoDef {
        static instance: ::protobuf::rt::LazyV2<SaveSliceInfoDef> = ::protobuf::rt::LazyV2::INIT;
        instance.get(SaveSliceInfoDef::new)
    }
}

impl ::protobuf::Clear for SaveSliceInfoDef {
    fn clear(&mut self) {
        self.full_name.clear();
        self.full_shape.clear();
        self.var_offset.clear();
        self.var_shape.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for SaveSliceInfoDef {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for SaveSliceInfoDef {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

#[derive(Clone,PartialEq,Eq,Debug,Hash)]
pub enum VariableSynchronization {
    VARIABLE_SYNCHRONIZATION_AUTO = 0,
    VARIABLE_SYNCHRONIZATION_NONE = 1,
    VARIABLE_SYNCHRONIZATION_ON_WRITE = 2,
    VARIABLE_SYNCHRONIZATION_ON_READ = 3,
}

impl ::protobuf::ProtobufEnum for VariableSynchronization {
    fn value(&self) -> i32 {
        *self as i32
    }

    fn from_i32(value: i32) -> ::std::option::Option<VariableSynchronization> {
        match value {
            0 => ::std::option::Option::Some(VariableSynchronization::VARIABLE_SYNCHRONIZATION_AUTO),
            1 => ::std::option::Option::Some(VariableSynchronization::VARIABLE_SYNCHRONIZATION_NONE),
            2 => ::std::option::Option::Some(VariableSynchronization::VARIABLE_SYNCHRONIZATION_ON_WRITE),
            3 => ::std::option::Option::Some(VariableSynchronization::VARIABLE_SYNCHRONIZATION_ON_READ),
            _ => ::std::option::Option::None
        }
    }

    fn values() -> &'static [Self] {
        static values: &'static [VariableSynchronization] = &[
            VariableSynchronization::VARIABLE_SYNCHRONIZATION_AUTO,
            VariableSynchronization::VARIABLE_SYNCHRONIZATION_NONE,
            VariableSynchronization::VARIABLE_SYNCHRONIZATION_ON_WRITE,
            VariableSynchronization::VARIABLE_SYNCHRONIZATION_ON_READ,
        ];
        values
    }

    fn enum_descriptor_static() -> &'static ::protobuf::reflect::EnumDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::EnumDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            ::protobuf::reflect::EnumDescriptor::new_pb_name::<VariableSynchronization>("VariableSynchronization", file_descriptor_proto())
        })
    }
}

impl ::std::marker::Copy for VariableSynchronization {
}

impl ::std::default::Default for VariableSynchronization {
    fn default() -> Self {
        VariableSynchronization::VARIABLE_SYNCHRONIZATION_AUTO
    }
}

impl ::protobuf::reflect::ProtobufValue for VariableSynchronization {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Enum(::protobuf::ProtobufEnum::descriptor(self))
    }
}

#[derive(Clone,PartialEq,Eq,Debug,Hash)]
pub enum VariableAggregation {
    VARIABLE_AGGREGATION_NONE = 0,
    VARIABLE_AGGREGATION_SUM = 1,
    VARIABLE_AGGREGATION_MEAN = 2,
    VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA = 3,
}

impl ::protobuf::ProtobufEnum for VariableAggregation {
    fn value(&self) -> i32 {
        *self as i32
    }

    fn from_i32(value: i32) -> ::std::option::Option<VariableAggregation> {
        match value {
            0 => ::std::option::Option::Some(VariableAggregation::VARIABLE_AGGREGATION_NONE),
            1 => ::std::option::Option::Some(VariableAggregation::VARIABLE_AGGREGATION_SUM),
            2 => ::std::option::Option::Some(VariableAggregation::VARIABLE_AGGREGATION_MEAN),
            3 => ::std::option::Option::Some(VariableAggregation::VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA),
            _ => ::std::option::Option::None
        }
    }

    fn values() -> &'static [Self] {
        static values: &'static [VariableAggregation] = &[
            VariableAggregation::VARIABLE_AGGREGATION_NONE,
            VariableAggregation::VARIABLE_AGGREGATION_SUM,
            VariableAggregation::VARIABLE_AGGREGATION_MEAN,
            VariableAggregation::VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA,
        ];
        values
    }

    fn enum_descriptor_static() -> &'static ::protobuf::reflect::EnumDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::EnumDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            ::protobuf::reflect::EnumDescriptor::new_pb_name::<VariableAggregation>("VariableAggregation", file_descriptor_proto())
        })
    }
}

impl ::std::marker::Copy for VariableAggregation {
}

impl ::std::default::Default for VariableAggregation {
    fn default() -> Self {
        VariableAggregation::VARIABLE_AGGREGATION_NONE
    }
}

impl ::protobuf::reflect::ProtobufValue for VariableAggregation {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Enum(::protobuf::ProtobufEnum::descriptor(self))
    }
}

static file_descriptor_proto_data: &'static [u8] = b"\
    \n(tensorflow/core/framework/variable.proto\x12\ntensorflow\"\xce\x03\n\
    \x0bVariableDef\x12#\n\rvariable_name\x18\x01\x20\x01(\tR\x0cvariableNam\
    e\x12,\n\x12initial_value_name\x18\x06\x20\x01(\tR\x10initialValueName\
    \x12)\n\x10initializer_name\x18\x02\x20\x01(\tR\x0finitializerName\x12#\
    \n\rsnapshot_name\x18\x03\x20\x01(\tR\x0csnapshotName\x12K\n\x13save_sli\
    ce_info_def\x18\x04\x20\x01(\x0b2\x1c.tensorflow.SaveSliceInfoDefR\x10sa\
    veSliceInfoDef\x12\x1f\n\x0bis_resource\x18\x05\x20\x01(\x08R\nisResourc\
    e\x12\x1c\n\ttrainable\x18\x07\x20\x01(\x08R\ttrainable\x12M\n\x0fsynchr\
    onization\x18\x08\x20\x01(\x0e2#.tensorflow.VariableSynchronizationR\x0f\
    synchronization\x12A\n\x0baggregation\x18\t\x20\x01(\x0e2\x1f.tensorflow\
    .VariableAggregationR\x0baggregation\"\x8a\x01\n\x10SaveSliceInfoDef\x12\
    \x1b\n\tfull_name\x18\x01\x20\x01(\tR\x08fullName\x12\x1d\n\nfull_shape\
    \x18\x02\x20\x03(\x03R\tfullShape\x12\x1d\n\nvar_offset\x18\x03\x20\x03(\
    \x03R\tvarOffset\x12\x1b\n\tvar_shape\x18\x04\x20\x03(\x03R\x08varShape*\
    \xac\x01\n\x17VariableSynchronization\x12!\n\x1dVARIABLE_SYNCHRONIZATION\
    _AUTO\x10\0\x12!\n\x1dVARIABLE_SYNCHRONIZATION_NONE\x10\x01\x12%\n!VARIA\
    BLE_SYNCHRONIZATION_ON_WRITE\x10\x02\x12$\n\x20VARIABLE_SYNCHRONIZATION_\
    ON_READ\x10\x03*\x9e\x01\n\x13VariableAggregation\x12\x1d\n\x19VARIABLE_\
    AGGREGATION_NONE\x10\0\x12\x1c\n\x18VARIABLE_AGGREGATION_SUM\x10\x01\x12\
    \x1d\n\x19VARIABLE_AGGREGATION_MEAN\x10\x02\x12+\n'VARIABLE_AGGREGATION_\
    ONLY_FIRST_REPLICA\x10\x03Bn\n\x18org.tensorflow.frameworkB\x0eVariableP\
    rotosP\x01Z=github.com/tensorflow/tensorflow/tensorflow/go/core/framewor\
    k\xf8\x01\x01b\x06proto3\
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

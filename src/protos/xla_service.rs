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
//! Generated file from `tensorflow/compiler/xla/rpc/xla_service.proto`

/// Generated files are compatible only with the same version
/// of protobuf runtime.
// const _PROTOBUF_VERSION_CHECK: () = ::protobuf::VERSION_2_17_0;

static file_descriptor_proto_data: &'static [u8] = b"\
    \n-tensorflow/compiler/xla/rpc/xla_service.proto\x12\x03xla\x1a!tensorfl\
    ow/compiler/xla/xla.proto2\xea\n\n\nXlaService\x12?\n\nUnregister\x12\
    \x16.xla.UnregisterRequest\x1a\x17.xla.UnregisterResponse\"\0\x12Q\n\x10\
    DeconstructTuple\x12\x1c.xla.DeconstructTupleRequest\x1a\x1d.xla.Deconst\
    ructTupleResponse\"\0\x123\n\x06Unpack\x12\x12.xla.UnpackRequest\x1a\x13\
    .xla.UnpackResponse\"\0\x129\n\x08GetShape\x12\x14.xla.GetShapeRequest\
    \x1a\x15.xla.GetShapeResponse\"\0\x12^\n\x18GetComputationGraphStats\x12\
    !.xla.ComputationGraphStatsRequest\x1a\x1d.xla.ComputationStatsResponse\
    \"\0\x129\n\x08LoadData\x12\x14.xla.LoadDataRequest\x1a\x15.xla.LoadData\
    Response\"\0\x12Q\n\x10TransferToClient\x12\x1c.xla.TransferToClientRequ\
    est\x1a\x1d.xla.TransferToClientResponse\"\0\x12Q\n\x10TransferToServer\
    \x12\x1c.xla.TransferToServerRequest\x1a\x1d.xla.TransferToServerRespons\
    e\"\0\x12Q\n\x10TransferToInfeed\x12\x1c.xla.TransferToInfeedRequest\x1a\
    \x1d.xla.TransferToInfeedResponse\"\0\x12Z\n\x13TransferFromOutfeed\x12\
    \x1f.xla.TransferFromOutfeedRequest\x1a\x20.xla.TransferFromOutfeedRespo\
    nse\"\0\x12B\n\x0bResetDevice\x12\x17.xla.ResetDeviceRequest\x1a\x18.xla\
    .ResetDeviceResponse\"\0\x12X\n\x14ComputeConstantGraph\x12\x20.xla.Comp\
    uteConstantGraphRequest\x1a\x1c.xla.ComputeConstantResponse\"\0\x12Q\n\
    \x10GetDeviceHandles\x12\x1c.xla.GetDeviceHandlesRequest\x1a\x1d.xla.Get\
    DeviceHandlesResponse\"\0\x12Z\n\x13CreateChannelHandle\x12\x1f.xla.Crea\
    teChannelHandleRequest\x1a\x20.xla.CreateChannelHandleResponse\"\0\x126\
    \n\x07Compile\x12\x13.xla.CompileRequest\x1a\x14.xla.CompileResponse\"\0\
    \x126\n\x07Execute\x12\x13.xla.ExecuteRequest\x1a\x14.xla.ExecuteRespons\
    e\"\0\x12X\n\x14ExecuteGraphParallel\x12\x20.xla.ExecuteGraphParallelReq\
    uest\x1a\x1c.xla.ExecuteParallelResponse\"\0\x12Q\n\x10WaitForExecution\
    \x12\x1c.xla.WaitForExecutionRequest\x1a\x1d.xla.WaitForExecutionRespons\
    e\"\0b\x06proto3\
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

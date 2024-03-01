// Automatically generated rust module for '_serialization.proto' file
// pb-rs _serialization.proto 

#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(unused_imports)]
#![allow(unknown_lints)]
#![allow(clippy::all)]
#![cfg_attr(rustfmt, rustfmt_skip)]


use std::borrow::Cow;
use std::collections::HashMap;
type KVMap<K, V> = HashMap<K, V>;
use quick_protobuf::{MessageInfo, MessageRead, MessageWrite, BytesReader, Writer, WriterBackend, Result};
use quick_protobuf::sizeofs::*;
use super::*;

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct Grammar<'a> {
    pub nodes: Vec<guidance::GrammarFunction<'a>>,
}

impl<'a> MessageRead<'a> for Grammar<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.nodes.push(r.read_message::<guidance::GrammarFunction>(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for Grammar<'a> {
    fn get_size(&self) -> usize {
        0
        + self.nodes.iter().map(|s| 1 + sizeof_len((s).get_size())).sum::<usize>()
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        for s in &self.nodes { w.write_with_tag(10, |w| w.write_message(s))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct EngineCallResponse<'a> {
    pub new_bytes: Cow<'a, [u8]>,
    pub is_generated: bool,
    pub new_bytes_prob: f32,
    pub capture_groups: KVMap<Cow<'a, str>, Cow<'a, str>>,
    pub capture_group_log_probs: KVMap<Cow<'a, str>, f32>,
    pub new_token_count: i32,
}

impl<'a> MessageRead<'a> for EngineCallResponse<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.new_bytes = r.read_bytes(bytes).map(Cow::Borrowed)?,
                Ok(16) => msg.is_generated = r.read_bool(bytes)?,
                Ok(29) => msg.new_bytes_prob = r.read_float(bytes)?,
                Ok(34) => {
                    let (key, value) = r.read_map(bytes, |r, bytes| Ok(r.read_string(bytes).map(Cow::Borrowed)?), |r, bytes| Ok(r.read_string(bytes).map(Cow::Borrowed)?))?;
                    msg.capture_groups.insert(key, value);
                }
                Ok(42) => {
                    let (key, value) = r.read_map(bytes, |r, bytes| Ok(r.read_string(bytes).map(Cow::Borrowed)?), |r, bytes| Ok(r.read_float(bytes)?))?;
                    msg.capture_group_log_probs.insert(key, value);
                }
                Ok(48) => msg.new_token_count = r.read_int32(bytes)?,
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for EngineCallResponse<'a> {
    fn get_size(&self) -> usize {
        0
        + if self.new_bytes == Cow::Borrowed(b"") { 0 } else { 1 + sizeof_len((&self.new_bytes).len()) }
        + if self.is_generated == false { 0 } else { 1 + sizeof_varint(*(&self.is_generated) as u64) }
        + if self.new_bytes_prob == 0f32 { 0 } else { 1 + 4 }
        + self.capture_groups.iter().map(|(k, v)| 1 + sizeof_len(2 + sizeof_len((k).len()) + sizeof_len((v).len()))).sum::<usize>()
        + self.capture_group_log_probs.iter().map(|(k, v)| 1 + sizeof_len(2 + sizeof_len((k).len()) + 4)).sum::<usize>()
        + if self.new_token_count == 0i32 { 0 } else { 1 + sizeof_varint(*(&self.new_token_count) as u64) }
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if self.new_bytes != Cow::Borrowed(b"") { w.write_with_tag(10, |w| w.write_bytes(&**&self.new_bytes))?; }
        if self.is_generated != false { w.write_with_tag(16, |w| w.write_bool(*&self.is_generated))?; }
        if self.new_bytes_prob != 0f32 { w.write_with_tag(29, |w| w.write_float(*&self.new_bytes_prob))?; }
        for (k, v) in self.capture_groups.iter() { w.write_with_tag(34, |w| w.write_map(2 + sizeof_len((k).len()) + sizeof_len((v).len()), 10, |w| w.write_string(&**k), 18, |w| w.write_string(&**v)))?; }
        for (k, v) in self.capture_group_log_probs.iter() { w.write_with_tag(42, |w| w.write_map(2 + sizeof_len((k).len()) + 4, 10, |w| w.write_string(&**k), 21, |w| w.write_float(*v)))?; }
        if self.new_token_count != 0i32 { w.write_with_tag(48, |w| w.write_int32(*&self.new_token_count))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct Byte<'a> {
    pub byte: Cow<'a, [u8]>,
    pub hidden: bool,
    pub commit_point: bool,
    pub nullable: bool,
    pub capture_name: Cow<'a, str>,
    pub temperature: f32,
}

impl<'a> MessageRead<'a> for Byte<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.byte = r.read_bytes(bytes).map(Cow::Borrowed)?,
                Ok(16) => msg.hidden = r.read_bool(bytes)?,
                Ok(24) => msg.commit_point = r.read_bool(bytes)?,
                Ok(32) => msg.nullable = r.read_bool(bytes)?,
                Ok(42) => msg.capture_name = r.read_string(bytes).map(Cow::Borrowed)?,
                Ok(53) => msg.temperature = r.read_float(bytes)?,
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for Byte<'a> {
    fn get_size(&self) -> usize {
        0
        + if self.byte == Cow::Borrowed(b"") { 0 } else { 1 + sizeof_len((&self.byte).len()) }
        + if self.hidden == false { 0 } else { 1 + sizeof_varint(*(&self.hidden) as u64) }
        + if self.commit_point == false { 0 } else { 1 + sizeof_varint(*(&self.commit_point) as u64) }
        + if self.nullable == false { 0 } else { 1 + sizeof_varint(*(&self.nullable) as u64) }
        + if self.capture_name == "" { 0 } else { 1 + sizeof_len((&self.capture_name).len()) }
        + if self.temperature == 0f32 { 0 } else { 1 + 4 }
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if self.byte != Cow::Borrowed(b"") { w.write_with_tag(10, |w| w.write_bytes(&**&self.byte))?; }
        if self.hidden != false { w.write_with_tag(16, |w| w.write_bool(*&self.hidden))?; }
        if self.commit_point != false { w.write_with_tag(24, |w| w.write_bool(*&self.commit_point))?; }
        if self.nullable != false { w.write_with_tag(32, |w| w.write_bool(*&self.nullable))?; }
        if self.capture_name != "" { w.write_with_tag(42, |w| w.write_string(&**&self.capture_name))?; }
        if self.temperature != 0f32 { w.write_with_tag(53, |w| w.write_float(*&self.temperature))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct ByteRange<'a> {
    pub byte_range: Cow<'a, [u8]>,
    pub hidden: bool,
    pub commit_point: bool,
    pub capture_name: Cow<'a, str>,
    pub temperature: f32,
}

impl<'a> MessageRead<'a> for ByteRange<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.byte_range = r.read_bytes(bytes).map(Cow::Borrowed)?,
                Ok(24) => msg.hidden = r.read_bool(bytes)?,
                Ok(32) => msg.commit_point = r.read_bool(bytes)?,
                Ok(42) => msg.capture_name = r.read_string(bytes).map(Cow::Borrowed)?,
                Ok(53) => msg.temperature = r.read_float(bytes)?,
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for ByteRange<'a> {
    fn get_size(&self) -> usize {
        0
        + if self.byte_range == Cow::Borrowed(b"") { 0 } else { 1 + sizeof_len((&self.byte_range).len()) }
        + if self.hidden == false { 0 } else { 1 + sizeof_varint(*(&self.hidden) as u64) }
        + if self.commit_point == false { 0 } else { 1 + sizeof_varint(*(&self.commit_point) as u64) }
        + if self.capture_name == "" { 0 } else { 1 + sizeof_len((&self.capture_name).len()) }
        + if self.temperature == 0f32 { 0 } else { 1 + 4 }
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if self.byte_range != Cow::Borrowed(b"") { w.write_with_tag(10, |w| w.write_bytes(&**&self.byte_range))?; }
        if self.hidden != false { w.write_with_tag(24, |w| w.write_bool(*&self.hidden))?; }
        if self.commit_point != false { w.write_with_tag(32, |w| w.write_bool(*&self.commit_point))?; }
        if self.capture_name != "" { w.write_with_tag(42, |w| w.write_string(&**&self.capture_name))?; }
        if self.temperature != 0f32 { w.write_with_tag(53, |w| w.write_float(*&self.temperature))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct Null { }

impl<'a> MessageRead<'a> for Null {
    fn from_reader(r: &mut BytesReader, _: &[u8]) -> Result<Self> {
        r.read_to_end();
        Ok(Self::default())
    }
}

impl MessageWrite for Null { }

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct ModelVariable<'a> {
    pub name: Cow<'a, str>,
    pub hidden: bool,
    pub commit_point: bool,
    pub capture_name: Cow<'a, str>,
    pub nullable: bool,
}

impl<'a> MessageRead<'a> for ModelVariable<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.name = r.read_string(bytes).map(Cow::Borrowed)?,
                Ok(16) => msg.hidden = r.read_bool(bytes)?,
                Ok(24) => msg.commit_point = r.read_bool(bytes)?,
                Ok(34) => msg.capture_name = r.read_string(bytes).map(Cow::Borrowed)?,
                Ok(40) => msg.nullable = r.read_bool(bytes)?,
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for ModelVariable<'a> {
    fn get_size(&self) -> usize {
        0
        + if self.name == "" { 0 } else { 1 + sizeof_len((&self.name).len()) }
        + if self.hidden == false { 0 } else { 1 + sizeof_varint(*(&self.hidden) as u64) }
        + if self.commit_point == false { 0 } else { 1 + sizeof_varint(*(&self.commit_point) as u64) }
        + if self.capture_name == "" { 0 } else { 1 + sizeof_len((&self.capture_name).len()) }
        + if self.nullable == false { 0 } else { 1 + sizeof_varint(*(&self.nullable) as u64) }
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if self.name != "" { w.write_with_tag(10, |w| w.write_string(&**&self.name))?; }
        if self.hidden != false { w.write_with_tag(16, |w| w.write_bool(*&self.hidden))?; }
        if self.commit_point != false { w.write_with_tag(24, |w| w.write_bool(*&self.commit_point))?; }
        if self.capture_name != "" { w.write_with_tag(34, |w| w.write_string(&**&self.capture_name))?; }
        if self.nullable != false { w.write_with_tag(40, |w| w.write_bool(*&self.nullable))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct Join<'a> {
    pub nullable: bool,
    pub values: Vec<i32>,
    pub name: Cow<'a, str>,
    pub hidden: bool,
    pub commit_point: bool,
    pub capture_name: Cow<'a, str>,
    pub max_tokens: i32,
}

impl<'a> MessageRead<'a> for Join<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(8) => msg.nullable = r.read_bool(bytes)?,
                Ok(18) => msg.values = r.read_packed(bytes, |r, bytes| Ok(r.read_int32(bytes)?))?,
                Ok(26) => msg.name = r.read_string(bytes).map(Cow::Borrowed)?,
                Ok(32) => msg.hidden = r.read_bool(bytes)?,
                Ok(40) => msg.commit_point = r.read_bool(bytes)?,
                Ok(50) => msg.capture_name = r.read_string(bytes).map(Cow::Borrowed)?,
                Ok(56) => msg.max_tokens = r.read_int32(bytes)?,
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for Join<'a> {
    fn get_size(&self) -> usize {
        0
        + if self.nullable == false { 0 } else { 1 + sizeof_varint(*(&self.nullable) as u64) }
        + if self.values.is_empty() { 0 } else { 1 + sizeof_len(self.values.iter().map(|s| sizeof_varint(*(s) as u64)).sum::<usize>()) }
        + if self.name == "" { 0 } else { 1 + sizeof_len((&self.name).len()) }
        + if self.hidden == false { 0 } else { 1 + sizeof_varint(*(&self.hidden) as u64) }
        + if self.commit_point == false { 0 } else { 1 + sizeof_varint(*(&self.commit_point) as u64) }
        + if self.capture_name == "" { 0 } else { 1 + sizeof_len((&self.capture_name).len()) }
        + if self.max_tokens == 0i32 { 0 } else { 1 + sizeof_varint(*(&self.max_tokens) as u64) }
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if self.nullable != false { w.write_with_tag(8, |w| w.write_bool(*&self.nullable))?; }
        w.write_packed_with_tag(18, &self.values, |w, m| w.write_int32(*m), &|m| sizeof_varint(*(m) as u64))?;
        if self.name != "" { w.write_with_tag(26, |w| w.write_string(&**&self.name))?; }
        if self.hidden != false { w.write_with_tag(32, |w| w.write_bool(*&self.hidden))?; }
        if self.commit_point != false { w.write_with_tag(40, |w| w.write_bool(*&self.commit_point))?; }
        if self.capture_name != "" { w.write_with_tag(50, |w| w.write_string(&**&self.capture_name))?; }
        if self.max_tokens != 0i32 { w.write_with_tag(56, |w| w.write_int32(*&self.max_tokens))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct Select<'a> {
    pub nullable: bool,
    pub values: Vec<i32>,
    pub name: Cow<'a, str>,
    pub hidden: bool,
    pub commit_point: bool,
    pub capture_name: Cow<'a, str>,
    pub max_tokens: i32,
    pub recursive: bool,
}

impl<'a> MessageRead<'a> for Select<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(8) => msg.nullable = r.read_bool(bytes)?,
                Ok(18) => msg.values = r.read_packed(bytes, |r, bytes| Ok(r.read_int32(bytes)?))?,
                Ok(26) => msg.name = r.read_string(bytes).map(Cow::Borrowed)?,
                Ok(32) => msg.hidden = r.read_bool(bytes)?,
                Ok(40) => msg.commit_point = r.read_bool(bytes)?,
                Ok(50) => msg.capture_name = r.read_string(bytes).map(Cow::Borrowed)?,
                Ok(56) => msg.max_tokens = r.read_int32(bytes)?,
                Ok(64) => msg.recursive = r.read_bool(bytes)?,
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for Select<'a> {
    fn get_size(&self) -> usize {
        0
        + if self.nullable == false { 0 } else { 1 + sizeof_varint(*(&self.nullable) as u64) }
        + if self.values.is_empty() { 0 } else { 1 + sizeof_len(self.values.iter().map(|s| sizeof_varint(*(s) as u64)).sum::<usize>()) }
        + if self.name == "" { 0 } else { 1 + sizeof_len((&self.name).len()) }
        + if self.hidden == false { 0 } else { 1 + sizeof_varint(*(&self.hidden) as u64) }
        + if self.commit_point == false { 0 } else { 1 + sizeof_varint(*(&self.commit_point) as u64) }
        + if self.capture_name == "" { 0 } else { 1 + sizeof_len((&self.capture_name).len()) }
        + if self.max_tokens == 0i32 { 0 } else { 1 + sizeof_varint(*(&self.max_tokens) as u64) }
        + if self.recursive == false { 0 } else { 1 + sizeof_varint(*(&self.recursive) as u64) }
    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        if self.nullable != false { w.write_with_tag(8, |w| w.write_bool(*&self.nullable))?; }
        w.write_packed_with_tag(18, &self.values, |w, m| w.write_int32(*m), &|m| sizeof_varint(*(m) as u64))?;
        if self.name != "" { w.write_with_tag(26, |w| w.write_string(&**&self.name))?; }
        if self.hidden != false { w.write_with_tag(32, |w| w.write_bool(*&self.hidden))?; }
        if self.commit_point != false { w.write_with_tag(40, |w| w.write_bool(*&self.commit_point))?; }
        if self.capture_name != "" { w.write_with_tag(50, |w| w.write_string(&**&self.capture_name))?; }
        if self.max_tokens != 0i32 { w.write_with_tag(56, |w| w.write_int32(*&self.max_tokens))?; }
        if self.recursive != false { w.write_with_tag(64, |w| w.write_bool(*&self.recursive))?; }
        Ok(())
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Default, PartialEq, Clone)]
pub struct GrammarFunction<'a> {
    pub function_type: guidance::mod_GrammarFunction::OneOffunction_type<'a>,
}

impl<'a> MessageRead<'a> for GrammarFunction<'a> {
    fn from_reader(r: &mut BytesReader, bytes: &'a [u8]) -> Result<Self> {
        let mut msg = Self::default();
        while !r.is_eof() {
            match r.next_tag(bytes) {
                Ok(10) => msg.function_type = guidance::mod_GrammarFunction::OneOffunction_type::join(r.read_message::<guidance::Join>(bytes)?),
                Ok(18) => msg.function_type = guidance::mod_GrammarFunction::OneOffunction_type::select(r.read_message::<guidance::Select>(bytes)?),
                Ok(26) => msg.function_type = guidance::mod_GrammarFunction::OneOffunction_type::byte(r.read_message::<guidance::Byte>(bytes)?),
                Ok(34) => msg.function_type = guidance::mod_GrammarFunction::OneOffunction_type::byte_range(r.read_message::<guidance::ByteRange>(bytes)?),
                Ok(42) => msg.function_type = guidance::mod_GrammarFunction::OneOffunction_type::model_variable(r.read_message::<guidance::ModelVariable>(bytes)?),
                Ok(t) => { r.read_unknown(bytes, t)?; }
                Err(e) => return Err(e),
            }
        }
        Ok(msg)
    }
}

impl<'a> MessageWrite for GrammarFunction<'a> {
    fn get_size(&self) -> usize {
        0
        + match self.function_type {
            guidance::mod_GrammarFunction::OneOffunction_type::join(ref m) => 1 + sizeof_len((m).get_size()),
            guidance::mod_GrammarFunction::OneOffunction_type::select(ref m) => 1 + sizeof_len((m).get_size()),
            guidance::mod_GrammarFunction::OneOffunction_type::byte(ref m) => 1 + sizeof_len((m).get_size()),
            guidance::mod_GrammarFunction::OneOffunction_type::byte_range(ref m) => 1 + sizeof_len((m).get_size()),
            guidance::mod_GrammarFunction::OneOffunction_type::model_variable(ref m) => 1 + sizeof_len((m).get_size()),
            guidance::mod_GrammarFunction::OneOffunction_type::None => 0,
    }    }

    fn write_message<W: WriterBackend>(&self, w: &mut Writer<W>) -> Result<()> {
        match self.function_type {            guidance::mod_GrammarFunction::OneOffunction_type::join(ref m) => { w.write_with_tag(10, |w| w.write_message(m))? },
            guidance::mod_GrammarFunction::OneOffunction_type::select(ref m) => { w.write_with_tag(18, |w| w.write_message(m))? },
            guidance::mod_GrammarFunction::OneOffunction_type::byte(ref m) => { w.write_with_tag(26, |w| w.write_message(m))? },
            guidance::mod_GrammarFunction::OneOffunction_type::byte_range(ref m) => { w.write_with_tag(34, |w| w.write_message(m))? },
            guidance::mod_GrammarFunction::OneOffunction_type::model_variable(ref m) => { w.write_with_tag(42, |w| w.write_message(m))? },
            guidance::mod_GrammarFunction::OneOffunction_type::None => {},
    }        Ok(())
    }
}

pub mod mod_GrammarFunction {

use super::*;

#[derive(Debug, PartialEq, Clone)]
pub enum OneOffunction_type<'a> {
    join(guidance::Join<'a>),
    select(guidance::Select<'a>),
    byte(guidance::Byte<'a>),
    byte_range(guidance::ByteRange<'a>),
    model_variable(guidance::ModelVariable<'a>),
    None,
}

impl<'a> Default for OneOffunction_type<'a> {
    fn default() -> Self {
        OneOffunction_type::None
    }
}

}


use crate::runtime;
use serde::{Deserialize, Serialize};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BlobId(u32);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum StorageOp {
    Set,
    Append,
}

#[allow(dead_code)]
pub mod bin_string {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &Vec<u8>, s: S) -> Result<S::Ok, S::Error> {
        let binstr = String::from_iter(v.iter().map(|b| *b as char));
        String::serialize(&binstr, s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let binstr = String::deserialize(d)?;
        Ok(binstr.chars().map(|c| c as u8).collect())
    }
}

pub mod hex_string {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use crate::bytes::{from_hex_string, to_hex_string};

    pub fn serialize<S: Serializer>(v: &Vec<u8>, s: S) -> Result<S::Ok, S::Error> {
        let hexstr = to_hex_string(v);
        String::serialize(&hexstr, s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let hexstr = String::deserialize(d)?;
        from_hex_string(&hexstr).map_err(serde::de::Error::custom)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum StorageCmd {
    /// Read variable. Returns StorageResp::ReadVar or StorageResp::VariableMissing.
    ReadVar { name: String },

    /// Write variable.
    /// If `when_version_is == None`, always writes the variable and returns StorageResp::WriteVar.
    /// Otherwise, if the variable has the specified version, it writes the variable
    /// and returns StorageResp::WriteVar.
    /// Otherwise (version conflict), returns either StorageResp::ReadVar or StorageResp::VariableMissing
    /// just like ReadVar would.
    WriteVar {
        name: String,
        #[serde(with = "hex_string")]
        value: Vec<u8>,
        op: StorageOp,
        when_version_is: Option<u64>,
    },
}

#[derive(Serialize, Deserialize, Debug)]
pub enum StorageResp {
    /// Upon handling the request the variable had the specified value and version number.
    ReadVar {
        version: u64,
        #[serde(with = "hex_string")]
        value: Vec<u8>,
    },
    /// Upon handling the request the variable was unset.
    VariableMissing {},
    /// The variable has been written, and the new version is returned.
    WriteVar { version: u64 },
}

// Public APIs

/// Stop the program - any error info is assumed to have been printed already.
pub fn aici_stop() -> ! {
    runtime::stop();
    panic!("didn't stop");
}

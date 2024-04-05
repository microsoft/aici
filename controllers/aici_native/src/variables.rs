use aici_abi::{StorageCmd, StorageOp, StorageResp};
use fxhash::FxHashMap;

#[derive(Default)]
pub struct Variables {
    pub variables: FxHashMap<String, (u64, Vec<u8>)>,
}

impl Variables {
    pub fn process_cmd(&mut self, cmd: StorageCmd) -> StorageResp {
        match cmd {
            StorageCmd::ReadVar { name } => match self.variables.get(&name).map(|x| x.clone()) {
                None => StorageResp::VariableMissing {},
                Some((version, value)) => StorageResp::ReadVar { value, version },
            },
            StorageCmd::WriteVar {
                name,
                value,
                when_version_is,
                op,
            } => {
                let curr = self.variables.get(&name).map(|x| x.clone());
                match curr {
                    Some((prev_version, prev_val)) => match when_version_is {
                        Some(v) if v != prev_version => StorageResp::ReadVar {
                            version: prev_version,
                            value: prev_val,
                        },
                        _ => {
                            let value = match op {
                                StorageOp::Append => {
                                    let mut v = prev_val.clone();
                                    v.extend(value);
                                    v
                                }
                                StorageOp::Set => value,
                            };
                            let version = prev_version + 1;
                            self.variables.insert(name, (version, value));
                            StorageResp::WriteVar { version }
                        }
                    },

                    None => match when_version_is {
                        None => {
                            self.variables.insert(name, (1, value));
                            StorageResp::WriteVar { version: 1 }
                        }
                        Some(_) => StorageResp::VariableMissing {},
                    },
                }
            }
        }
    }
}


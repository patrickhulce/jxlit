use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub fn decode(input: Buffer) -> Result<Buffer> {
    Ok(Buffer::from(jxlit::decode(input.as_ref())))
}

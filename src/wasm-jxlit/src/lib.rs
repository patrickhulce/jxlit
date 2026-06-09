use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn decode(input: &[u8]) -> Vec<u8> {
    jxlit::decode(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_returns_empty_buffer() {
        assert!(decode(b"not-a-jxl").is_empty());
    }
}

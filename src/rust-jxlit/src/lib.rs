pub fn decode(input: &[u8]) -> Vec<u8> {
    let _ = input;
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_returns_empty_buffer() {
        assert!(decode(b"not-a-jxl").is_empty());
    }
}

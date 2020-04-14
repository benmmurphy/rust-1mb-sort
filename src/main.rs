extern crate rand;
use std::io::BufRead;

struct Sorter {
    compressed_storage: CircularBitBuffer,
    compressed_length: usize,
    int_buffer: Vec<u32>,
}

impl Sorter {
    fn new() -> Sorter {
        Sorter {
            compressed_storage: CircularBitBuffer::new(1_011_725 * 8),
            compressed_length: 0,
            int_buffer: Vec::with_capacity(8192),
        }
    }

    fn add_value(self: &mut Self, value: u32) {
        self.int_buffer.push(value);
        if self.int_buffer.len() == self.int_buffer.capacity() {
            self.flush();
        }
    }

    fn finalize(self: &mut Self) {
        if self.int_buffer.is_empty() {
            return;
        }

        self.flush();
    }

    fn flush(self: &mut Self) {
        self.int_buffer.sort_unstable();

        let mut delta_decoder = DeltaDecoder::new();
        let mut delta_encoder = DeltaEncoder::new();

        let mut buffer_position = 0;
        let mut compressed_position = 0;

        let mut maybe_next_compressed = None;

        let mut storage = LimitedCircularBuffer::new(&mut self.compressed_storage);
        /* the decoder will overrun the read buffer which is quite bad because we only have 1 mark so it will start reading
        bytes we have written. the limited circular buffer snapshots the readable bytes and won't let us read more
        than those bytes */

        while compressed_position < self.compressed_length
            && buffer_position < self.int_buffer.len()
        {
            maybe_next_compressed =
                maybe_next_compressed.or_else(|| Some(delta_decoder.decode_int(&mut storage)));

            let next_compressed = maybe_next_compressed.unwrap();

            if next_compressed <= self.int_buffer[buffer_position] {
                delta_encoder.encode_int(next_compressed, &mut storage);
                compressed_position += 1;
                maybe_next_compressed = None;
            } else {
                /* TODO: potentially unsafe because we could be writing 8192 ints to the front of the buffer
                and only removing 1 value from the front */
                delta_encoder.encode_int(self.int_buffer[buffer_position], &mut storage);
                buffer_position += 1;
            }
        }

        while compressed_position < self.compressed_length {
            maybe_next_compressed =
                maybe_next_compressed.or_else(|| Some(delta_decoder.decode_int(&mut storage)));
            let next_compressed = maybe_next_compressed.unwrap();

            delta_encoder.encode_int(next_compressed, &mut storage);
            compressed_position += 1;
            maybe_next_compressed = None;
        }

        while buffer_position < self.int_buffer.len() {
            delta_encoder.encode_int(self.int_buffer[buffer_position], &mut storage);
            buffer_position += 1;
        }

        self.compressed_length += self.int_buffer.len();
        self.int_buffer.clear();
        delta_encoder.flush(&mut storage);
    }

    fn read_sorted<F: FnMut(u32)>(self: &mut Self, mut callback: F) {
        let mut delta_decoder = DeltaDecoder::new();
        for _i in 0..self.compressed_length {
            callback(delta_decoder.decode_int(&mut self.compressed_storage));
        }

        self.compressed_length = 0;
    }
}

trait CircularBuffer {
    fn read_bit(self: &mut Self) -> Option<bool>;
    fn write_bit(self: &mut Self, b: bool) -> bool;
}

struct LimitedCircularBuffer<'a> {
    buffer: &'a mut CircularBitBuffer,
    size: usize,
}

impl<'a> LimitedCircularBuffer<'a> {
    fn new(buffer: &'a mut CircularBitBuffer) -> LimitedCircularBuffer<'a> {
        let size = buffer.size;

        LimitedCircularBuffer { buffer, size }
    }
}

impl<'a> CircularBuffer for LimitedCircularBuffer<'a> {
    fn read_bit(self: &mut Self) -> Option<bool> {
        if self.size == 0 {
            return None;
        }

        self.size -= 1;

        self.buffer.read_bit()
    }

    fn write_bit(self: &mut Self, b: bool) -> bool {
        self.buffer.write_bit(b)
    }
}

#[derive(Debug, Clone)]
struct CircularBitBuffer {
    buffer: Vec<u32>,
    read_pos: usize,
    write_pos: usize,

    size: usize,
}

const BIT_BUFFER_ELEMENT_SIZE: usize = 32;

impl CircularBitBuffer {
    fn new(bit_size: usize) -> CircularBitBuffer {
        let element_size = (bit_size + BIT_BUFFER_ELEMENT_SIZE - 1) / BIT_BUFFER_ELEMENT_SIZE;

        CircularBitBuffer {
            buffer: vec![0; element_size],
            read_pos: 0,
            write_pos: 0,
            size: 0,
        }
    }

    fn get_bits(self: &Self) -> String {
        let mut copy = self.clone();
        let mut result = String::new();
        while copy.size > 0 {
            if copy.read_bit().unwrap() {
                result.push('1');
            } else {
                result.push('0');
            }
        }

        result
    }

    fn increment_write_position(self: &mut Self) {
        self.write_pos += 1;
        if self.write_pos >= self.buffer.len() * BIT_BUFFER_ELEMENT_SIZE {
            self.write_pos = 0;
        }
    }

    fn increment_read_position(self: &mut Self) {
        self.read_pos += 1;
        if self.read_pos >= self.buffer.len() * BIT_BUFFER_ELEMENT_SIZE {
            self.read_pos = 0;
        }
    }
}

impl CircularBuffer for CircularBitBuffer {
    fn read_bit(self: &mut Self) -> Option<bool> {
        if self.size == 0 {
            return None;
        }
        let byte = self.buffer[self.read_pos / BIT_BUFFER_ELEMENT_SIZE];

        self.size -= 1;

        let bit_pos = 1 << (self.read_pos % BIT_BUFFER_ELEMENT_SIZE);

        let result = Some((bit_pos & byte) == bit_pos);

        self.increment_read_position();

        result
    }

    fn write_bit(self: &mut Self, bit: bool) -> bool {
        if self.size >= self.buffer.len() * BIT_BUFFER_ELEMENT_SIZE {
            return false;
        }

        let write_byte = self.write_pos / BIT_BUFFER_ELEMENT_SIZE;

        let byte = self.buffer[write_byte];
        let mask = 1 << (self.write_pos % BIT_BUFFER_ELEMENT_SIZE);
        let inv = (-(bit as i32)) as u32;

        let new_byte = (byte & !mask) | (inv & mask);

        self.buffer[write_byte] = new_byte;
        self.size += 1;
        self.increment_write_position();

        true
    }
}
struct ARDecodeState {
    frac: u32,
    bits: u32,
    range: Range,
}

const ENCODINGS: [u32; 128] = [
    0x00000000, 0x0288df0d, 0x050b5170, 0x07876772, 0x09fd3132, 0x0c6cbea6, 0x0ed61f9e, 0x113963bd,
    0x13969a84, 0x15edd349, 0x183f1d3c, 0x1a8a8767, 0x1cd020ad, 0x1f0ff7cd, 0x214a1b5e, 0x237e99d5,
    0x25ad817f, 0x27d6e088, 0x29fac4f7, 0x2c193cae, 0x2e32556d, 0x30461cd2, 0x3254a056, 0x345ded53,
    0x366210ff, 0x3861186f, 0x3a5b1097, 0x3c50064a, 0x3e40063a, 0x402b1cfa, 0x421156fd, 0x43f2c096,
    0x45cf65f7, 0x47a75337, 0x497a944b, 0x4b49350b, 0x4d134131, 0x4ed8c45a, 0x5099ca03, 0x52565d8f,
    0x540e8a41, 0x55c25b44, 0x5771dba1, 0x591d1649, 0x5ac41611, 0x5c66e5b1, 0x5e058fc6, 0x5fa01ed4,
    0x61369d42, 0x62c9155d, 0x6457915a, 0x65e21b52, 0x6768bd44, 0x68eb8119, 0x6a6a709d, 0x6be59584,
    0x6d5cf96d, 0x6ed0a5d9, 0x7040a435, 0x71acfdd4, 0x7315bbf3, 0x747ae7b7, 0x75dc8a2c, 0x773aac4a,
    0x789556f1, 0x79ec92ea, 0x7b4068e8, 0x7c90e188, 0x7dde0551, 0x7f27dcb6, 0x806e7012, 0x81b1c7ad,
    0x82f1ebb8, 0x842ee451, 0x8568b981, 0x869f733b, 0x87d31961, 0x8903b3be, 0x8a314a0b, 0x8b5be3ec,
    0x8c8388f4, 0x8da840a1, 0x8eca125e, 0x8fe90583, 0x91052157, 0x921e6d0d, 0x9334efc6, 0x9448b091,
    0x9559b66c, 0x96680843, 0x9773acef, 0x987cab39, 0x998309d8, 0x9a86cf73, 0x9b8802a0, 0x9c86a9e3,
    0x9d82cbb1, 0x9e7c6e6d, 0x9f73986c, 0xa0684ff2, 0xa15a9b31, 0xa24a804f, 0xa3380560, 0xa423306a,
    0xa50c0761, 0xa5f2902e, 0xa6d6d0a7, 0xa7b8ce96, 0xa8988fb6, 0xa97619b2, 0xaa517227, 0xab2a9ea6,
    0xac01a4ae, 0xacd689b4, 0xada9531d, 0xae7a0640, 0xaf48a868, 0xb0153ed2, 0xb0dfcead, 0xb1a85d1a,
    0xb26eef31, 0xb33389f9, 0xb3f6326e, 0xb4b6ed7f, 0xb575c00e, 0xb632aef2, 0xb6edbef5, 0xb7a6f4d4,
];

impl ARDecodeState {
    fn new() -> ARDecodeState {
        ARDecodeState {
            frac: 0,
            bits: 0,
            range: Range::all(),
        }
    }

    fn next_bit<B: CircularBuffer>(vec: &mut B) -> u32 {
        match vec.read_bit() {
            Some(bit) => {
                if bit {
                    1
                } else {
                    0
                }
            }

            None => 0,
        }
    }

    fn decode_delta<B: CircularBuffer>(self: &mut Self, reader: &mut B) -> u32 {
        let mut delta = 0;

        loop {
            let next_delta = self.decode(reader);
            delta += next_delta;

            if next_delta != ENCODINGS.len() as u32 - 1 {
                return delta;
            }
        }
    }

    fn decode<B: CircularBuffer>(self: &mut Self, reader: &mut B) -> u32 {
        while self.bits < 32 {
            let bit = ARDecodeState::next_bit(reader);
            self.frac = (self.frac << 1) | bit;
            self.bits += 1;
        }

        let index =
            /* its possible to remove a lerp because we redo self.range.lerp() in self.range.lerp_low() in narrow() */
            match ENCODINGS.binary_search_by(|probe| self.range.lerp(*probe).cmp(&self.frac)) {
                Ok(index) => index,

                Err(index) => {
                    if index == 0 {
                        panic!("0 index???");
                    }
                    index - 1
                }
            };
        /* min_value >= ENCODINGS[index] && min_value < ENCODINGS[index + 1] */

        self.narrow(index as u32, reader);
        index as u32
    }

    fn narrow<B: CircularBuffer>(self: &mut Self, v: u32, reader: &mut B) {
        let mut new_low = self.range.lerp_low(v);
        let mut new_high = self.range.lerp_high(v);

        loop {
            if new_high < TOP_BIT {
                /* nothing*/
            } else if new_low >= TOP_BIT {
                self.frac -= TOP_BIT;
                new_low -= TOP_BIT;
                new_high -= TOP_BIT;
            } else if new_low >= SECOND_BIT && new_high < (TOP_BIT | SECOND_BIT) {
                self.frac -= SECOND_BIT;
                new_low -= SECOND_BIT;
                new_high -= SECOND_BIT;
            } else {
                break;
            }

            new_low <<= 1;
            new_high = (new_high << 1) | 1;
            self.frac = (self.frac << 1) | ARDecodeState::next_bit(reader);
        }

        self.range = Range::new(new_low, new_high);
    }
}

struct AREncodeState {
    range: Range,
    carry_bits: u32,
}

const TOP_BIT: u32 = (1 << 31);
const SECOND_BIT: u32 = (1 << 30);

#[derive(Debug)]
struct Range {
    /* range is inclusive in the lower bound
       the upperbound is inclusive but extended by 1s infinitely to the right
       so 0xFF_FF_FF_FF is equivalent to 0.111111.... which is equal to 1.0
    */
    low: u32,
    high: u32,
}

impl Range {
    fn lerp(self: &Self, fraction: u32) -> u32 {
        /*
           n/(2^32) * y (2^32) => (n*y) / (2^64)
           [then we can throw away the lower bits and just keep the top part of the fraction]

           technically the size of the range is (high - low + 1) because we extend high infinitely
           to the right with 1s so eventually it becomes high + 1.

           note: we never round results upwards because this could cause weird stuff where the expected
           limit is <= high but then becomes high + 1 and crosses into the next boundary
        */
        let range = self.high as u64 - self.low as u64 + 1;

        self.low + ((fraction as u64 * range) >> 32) as u32
    }

    fn lerp_low(self: &Self, codepoint: u32) -> u32 {
        self.lerp(ENCODINGS[codepoint as usize])
    }

    fn lerp_high(self: &Self, codepoint: u32) -> u32 {
        if codepoint + 1 == ENCODINGS.len() as u32 {
            self.lerp(0xFF_FF_FF_FF)
        } else {
            /* the new high is less than the next values low boundary:
               ex:
               0x0 -> 0xdeadbeee .. ffffffff
               0xdeadbeef -> 0xffffffff .. ffffff
            */
            self.lerp(ENCODINGS[(codepoint + 1) as usize]) - 1
        }
    }

    fn all() -> Range {
        Range {
            low: 0,
            high: 0xFF_FF_FF_FF,
        }
    }

    fn new(low: u32, high: u32) -> Range {
        Range { low, high }
    }
}

struct DeltaDecoder {
    last: u32,
    decoder: ARDecodeState,
}

impl DeltaDecoder {
    fn new() -> DeltaDecoder {
        DeltaDecoder {
            last: 0,
            decoder: ARDecodeState::new(),
        }
    }

    fn decode_int<B: CircularBuffer>(self: &mut Self, vec: &mut B) -> u32 {
        let delta = self.decoder.decode_delta(vec);
        let current = self.last + delta;
        self.last = current;
        current
    }
}

struct DeltaEncoder {
    last: u32,
    encoder: AREncodeState,
}

impl DeltaEncoder {
    fn new() -> DeltaEncoder {
        DeltaEncoder {
            last: 0,
            encoder: AREncodeState::new(),
        }
    }

    fn encode_int<B: CircularBuffer>(self: &mut Self, value: u32, vec: &mut B) {
        let delta = value - self.last;
        self.encoder.encode_delta(delta, vec);
        self.last = value;
    }

    fn flush<B: CircularBuffer>(self: &mut Self, vec: &mut B) {
        self.encoder.flush(vec);
    }
}

impl AREncodeState {
    fn new() -> AREncodeState {
        AREncodeState {
            range: Range::all(),
            carry_bits: 0,
        }
    }

    fn flush<B: CircularBuffer>(self: &mut AREncodeState, vec: &mut B) {
        /*
        The general rule for the last two bits given range of [0, 64) is:
        If the source interval contains the second quadrant [16,32), output 01 plus delayed bits 1’s.
        If the source interval contains the third quadrant [32, 48), output 10 plus delayed bits 0’s.
        */

        /* pre-condition:
           TOP_BIT_high != TOP_BIT_low
           ! (SECOND_BIT_high == 0 && SECOND_BIT_low == 1)
           high > low

           possible outcomes:

           00, 00 [not possible]
           00, 01 [not possible]
           00, 10 [possible] [<16, >32 and <48] contains second quadrant, may contain third quadrant but not guaranteed (ie: could be 0-47).
           00, 11 [possible] [<16, >= 48] contains second and third quadrant

           [outcomes above low < SECOND_BIT and always contain second quadrant]

           01, 00 [not possible]
           01, 01 [not possible]
           01, 10 [not possible]
           01, 11 [possible] [>=16 and <32, >=48] may contain second quadrant but not guaranteed. (ie: could be 16-48, but also could be 18-48). guaranteed to contain third quadrant.

           [outcomes above low >= SECOND_BIT and always contain third quadrant]
           10, 00 [not possible]
           10, 01 [not possible]
           10, 10 [not possible]
           10, 11 [not possible]

           11, 00 [not possible]
           11, 01 [not possible]
           11, 10 [not possible]
           11, 11 [not possible]
        */

        self.carry_bits += 1;
        if self.range.low < SECOND_BIT {
            self.output_bit(false, vec);
        } else {
            self.output_bit(true, vec);
        }
    }

    fn output_bit<B: CircularBuffer>(self: &mut Self, bit: bool, vec: &mut B) {
        if !vec.write_bit(bit) {
            panic!("buffer_out_of_space");
        }
        while self.carry_bits > 0 {
            if !vec.write_bit(!bit) {
                panic!("buffer_out_of_space");
            }
            self.carry_bits -= 1;
        }
    }

    fn encode_delta<B: CircularBuffer>(self: &mut Self, mut delta: u32, vec: &mut B) {
        loop {
            if delta < ENCODINGS.len() as u32 - 1 {
                self.ar_encode(delta, vec);
                return;
            }

            self.ar_encode(ENCODINGS.len() as u32 - 1, vec);
            delta -= ENCODINGS.len() as u32 - 1;
        }
    }

    fn ar_encode<B: CircularBuffer>(self: &mut Self, v: u32, vec: &mut B) {
        let mut new_low = self.range.lerp_low(v);
        let mut new_high = self.range.lerp_high(v);

        loop {
            if new_high < TOP_BIT {
                self.output_bit(false, vec);
            } else if new_low >= TOP_BIT {
                self.output_bit(true, vec);
                new_low -= TOP_BIT;
                new_high -= TOP_BIT;
            } else if new_low >= SECOND_BIT && new_high < (SECOND_BIT | TOP_BIT) {
                self.carry_bits += 1;
                new_low -= SECOND_BIT;
                new_high -= SECOND_BIT;
            } else {
                break;
            }

            new_low <<= 1;
            new_high = (new_high << 1) | 1;
        }

        let range_size = new_high - new_low;
        if range_size < SECOND_BIT {
            panic!("range_underflow");
        }

        self.range = Range::new(new_low, new_high);
    }
}

fn main() {
    let stdin = std::io::stdin();
    let locked = stdin.lock();
    let mut sorter = Sorter::new();

    for maybe_line in locked.lines() {
        let line = maybe_line.unwrap();
        let v: u32 = line.parse().unwrap();
        sorter.add_value(v);
    }

    sorter.finalize();

    sorter.read_sorted(|v| {
        println!("{}", v);
    });
}

#[cfg(test)]
mod tests {
    use crate::rand::Rng;
    use crate::CircularBuffer;

    #[test]
    fn test_circular_bit_buffer_wrap() {
        let mut buffer = super::CircularBitBuffer::new(2 * 8);
        let v = 0b1000001111000110;

        for _i in 0..4 {
            assert_eq!(true, buffer.write_bit(true));
        }

        for _i in 0..4 {
            assert_eq!(true, buffer.read_bit().unwrap());
        }

        for i in 0..16 {
            let bit = (v >> i) & 1;
            assert_eq!(true, buffer.write_bit(bit == 1));
        }

        for i in 0..16 {
            let bit = if buffer.read_bit().unwrap() { 1 } else { 0 };
            assert_eq!((v >> i) & 1, bit);
        }
    }

    #[test]
    fn test_circular_bit_buffer() {
        let mut buffer = super::CircularBitBuffer::new(2 * 8);
        let v = 0b1000001111000110;

        for i in 0..16 {
            let bit = (v >> i) & 1;
            assert_eq!(true, buffer.write_bit(bit == 1));
        }

        for i in 0..16 {
            let bit = if buffer.read_bit().unwrap() { 1 } else { 0 };
            assert_eq!((v >> i) & 1, bit, "mismatch on {}", i);
        }
    }

    fn check_seq(seq: &[u32]) {
        let mut vec = super::CircularBitBuffer::new(1_011_725 * 8);
        let mut delta_encoder = super::DeltaEncoder::new();

        for v in seq {
            delta_encoder.encode_int(*v, &mut vec);
        }

        delta_encoder.flush(&mut vec);

        println!(
            "SIZE: {} {} {}",
            vec.size,
            seq.len(),
            vec.size as f64 / seq.len() as f64
        );

        let mut delta_decoder = super::DeltaDecoder::new();
        let mut decoded = Vec::new();

        for _v in seq {
            decoded.push(delta_decoder.decode_int(&mut vec));
        }

        assert_eq!(decoded, seq);
    }

    #[test]
    fn test_simple_encoding() {
        check_seq(&[0, 128, 256]);
        check_seq(&[0, 1, 2, 3, 4, 4, 4]);

        let start = 0;
        check_seq(&[start, start + 144, start + 144 + 1707]);
    }

    #[test]
    fn test_ar_encoder() {
        let mut encoder = super::AREncodeState::new();
        let mut buffer = super::CircularBitBuffer::new(1_011_725 * 8);

        encoder.ar_encode(18, &mut buffer);

        let n = 787_401 - 90;
        let other_n = 90;

        let highest = 127;

        for _i in 0..n {
            encoder.ar_encode(highest, &mut buffer);
        }

        encoder.ar_encode(highest - 1, &mut buffer);

        for _i in 0..other_n {
            encoder.ar_encode(highest, &mut buffer);
        }

        encoder.ar_encode(6, &mut buffer);

        encoder.flush(&mut buffer);

        let mut decoder = super::ARDecodeState::new();

        assert_eq!(18, decoder.decode(&mut buffer));

        for i in 0..n {
            assert_eq!(highest, decoder.decode(&mut buffer), "mismatch on {}", i);
        }

        assert_eq!(highest - 1, decoder.decode(&mut buffer));

        for i in 0..other_n {
            assert_eq!(
                highest,
                decoder.decode(&mut buffer),
                "mismatch on second {}",
                i
            );
        }

        assert_eq!(6, decoder.decode(&mut buffer));
    }

    #[test]
    fn test_encodes() {
        let mut rng = rand::thread_rng();
        for _i in 0..10 {
            println!("CHECKING FULL SORT...");
            let n = 1_000_000;
            let mut values: Vec<u32> = (0..n).map(|_| rng.gen_range(0, 1_000_000_00)).collect();
            values.sort();

            check_seq(&values);
        }
    }

    #[test]
    fn test_sorter() {
        let mut rng = rand::thread_rng();
        for _i in 0..1 {
            let n = 1_000_000;
            let mut values: Vec<u32> = (0..n).map(|_| rng.gen_range(0, 1_000_000_00)).collect();

            let mut sorter = super::Sorter::new();
            for v in &values {
                sorter.add_value(*v);
            }

            sorter.finalize();

            let mut final_sorted = Vec::new();

            sorter.read_sorted(|v| final_sorted.push(v));

            values.sort();

            assert_eq!(final_sorted, values);
        }
    }
}

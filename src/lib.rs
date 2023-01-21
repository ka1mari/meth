#![feature(const_trait_impl)]
#![feature(portable_simd)]
//#![no_std]

use core::mem;

pub use crate::mixed_slice::MixedSliceExt;
pub use crate::scalar_slice::ScalarSliceExt;
pub use crate::simd_slice::SimdSliceExt;

mod mixed_slice;
mod scalar_slice;
mod simd_slice;

/// Determine the nearest supported lane size for the provided element.
pub const fn determine_lanes<T>(mut lanes: usize) -> usize {
    let size = mem::size_of::<T>();

    // this will be optimized out as `size` is a constant
    assert!(size <= 64, "up to 512-byte vectors are supported");

    // clamp to the maximum supported vector size
    lanes = lanes.min(512 / size);

    // check if it's already a valid lane size
    if lanes.is_power_of_two() {
        lanes
    } else {
        lanes.next_power_of_two() >> 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanes() {
        assert_eq!(determine_lanes::<u8>(0), 0);
        assert_eq!(determine_lanes::<u8>(1), 1);

        assert_eq!(determine_lanes::<u8>(2), 2);
        assert_eq!(determine_lanes::<u8>(3), 2);

        assert_eq!(determine_lanes::<u8>(4), 4);
        assert_eq!(determine_lanes::<u8>(5), 4);
        assert_eq!(determine_lanes::<u8>(6), 4);
        assert_eq!(determine_lanes::<u8>(7), 4);

        assert_eq!(determine_lanes::<u64>(0), 0);
        assert_eq!(determine_lanes::<u64>(1), 1);
    }
}

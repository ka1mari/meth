use crate::SimdSliceExt;
use core::{
    iter, ops,
    simd::{LaneCount, Simd, SimdElement, SupportedLaneCount},
};

macro_rules! mixed_slice_ext {
    ($($fn:ident, $trait:ident, $simd:ident, $op:tt;)*) => {
        /// Extension trait for slices of scalars which could use SIMD.
        pub trait MixedSliceExt<T> {
            $(
                fn $fn<const LANES: usize>(&mut self, other: &[T])
                where
                    T: ops::$trait + SimdElement,
                    Simd<T, LANES>: ops::$trait,
                    LaneCount<LANES>: SupportedLaneCount;
            )*
        }

        impl<T> MixedSliceExt<T> for [T]
        where
            T: std::fmt::Debug,
        {
            $(
                #[inline]
                fn $fn<const LANES: usize>(&mut self, other: &[T])
                where
                    T: ops::$trait + SimdElement,
                    Simd<T, LANES>: ops::$trait,
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    assert_eq!(self.len(), other.len(), "slices must have the same length");

                    let (self_prefix, self_middle, self_suffix) = self.as_simd_mut::<LANES>();
                    let (other_prefix, other_middle, other_suffix) = other.as_simd::<LANES>();

                    // input slices have the same length, but their addresses may have different
                    // allignment, so `ScalarSliceExt` cannot be used on the prefix and suffix.
                    let self_edges = self_prefix.iter_mut().chain(self_suffix.iter_mut());
                    let other_edges = other_prefix.iter().chain(other_suffix.iter());

                    for (x, y) in iter::zip(self_edges, other_edges) {
                        *x $op *y;
                    }

                    self_middle.$simd(other_middle);
                }
            )*
        }
    }
}

mixed_slice_ext! {
    mixed_add, AddAssign, simd_add, +=;
    mixed_div, DivAssign, simd_div, /=;
    mixed_mul, MulAssign, simd_mul, *=;
    mixed_rem, RemAssign, simd_rem, %=;
    mixed_sub, SubAssign, simd_sub, -=;

    mixed_bitand, BitAndAssign, simd_bitand, &=;
    mixed_bitor, BitOrAssign, simd_bitor, |=;
    mixed_bitxor, BitXorAssign, simd_bitxor, ^=;

    mixed_shl, ShlAssign, simd_shl, <<=;
    mixed_shr, ShrAssign, simd_shr, >>=;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_ops() {
        let mut x = [1.0_f32; 15];
        let y = [2.0_f32; 15];
        let z = [3.0_f32; 15];

        x.mixed_add::<8>(&y);

        assert_eq!(x, z);
    }
}

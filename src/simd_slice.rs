use core::{
    iter, ops,
    simd::{LaneCount, Simd, SimdElement, SupportedLaneCount},
};

macro_rules! simd_slice_ext {
    ($($fn:ident, $trait:ident, $op:tt;)*) => {
        /// Extension trait for slices of SIMD vectors.
        pub trait SimdSliceExt<T, const LANES: usize>
        where
            T: SimdElement,
            LaneCount<LANES>: SupportedLaneCount,
        {
            $(
                fn $fn(&mut self, other: &[Simd<T, LANES>])
                where
                    Simd<T, LANES>: ops::$trait;
            )*
        }

        impl<T, const LANES: usize> SimdSliceExt<T, LANES> for [Simd<T, LANES>]
        where
            T: SimdElement,
            LaneCount<LANES>: SupportedLaneCount,
        {
            $(
                #[inline]
                fn $fn(&mut self, other: &[Simd<T, LANES>])
                where
                    Simd<T, LANES>: ops::$trait,
                {
                    assert_eq!(self.len(), other.len(), "slices must have the same length");

                    for (x, y) in iter::zip(self.iter_mut(), other.iter()) {
                        *x $op *y;
                    }
                }
            )*
        }
    }
}

simd_slice_ext! {
    simd_add, AddAssign, +=;
    simd_div, DivAssign, /=;
    simd_mul, MulAssign, *=;
    simd_rem, RemAssign, %=;
    simd_sub, SubAssign, -=;

    simd_bitand, BitAndAssign, &=;
    simd_bitor, BitOrAssign, |=;
    simd_bitxor, BitXorAssign, ^=;

    simd_shl, ShlAssign, <<=;
    simd_shr, ShrAssign, >>=;
}

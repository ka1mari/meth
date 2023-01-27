use crate::{determine_lanes, MixedSliceExt};
use core::{
    ops, hint,
    simd::{LaneCount, Simd, SimdElement, SupportedLaneCount},
};

macro_rules! slice_ext {
    ($($fn:ident, $trait:ident, $mixed:ident;)*) => {
        /// Extension trait for slices of scalars which could use SIMD.
        pub trait SliceExt<T> {
            $(
                fn $fn<const LANES: usize>(&mut self, other: &[T])
                where
                    T: ops::$trait + SimdElement,
                    Simd<T, LANES>: ops::$trait,
                    LaneCount<LANES>: SupportedLaneCount;
            )*
        }

        impl<T> SliceExt<T> for [T] {
            $(
                #[inline]
                fn $fn<const LANES: usize>(&mut self, other: &[T])
                where
                    T: ops::$trait + SimdElement,
                    Simd<T, LANES>: ops::$trait,
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    assert_eq!(self.len(), other.len(), "slices must have the same length");

                    let lanes = determine_lanes::<T>(self.len());

                    match lanes {
                        0 => {},
                        1 => self.$mixed(other),
                        2 => self.$mixed(other),
                        4 => self.$mixed(other),
                        8 => self.$mixed(other),
                        16 => self.$mixed(other),
                        32 => self.$mixed(other),
                        64 => self.$mixed(other),
                        _ => unsafe { hint::unreachable_unchecked() },
                    }
                }
            )*
        }
    }
}

slice_ext! {
    add, AddAssign, mixed_add;
    div, DivAssign, mixed_div;
    mul, MulAssign, mixed_mul;
    rem, RemAssign, mixed_rem;
    sub, SubAssign, mixed_sub;

    bitand, BitAndAssign, mixed_bitand;
    bitor, BitOrAssign, mixed_bitor;
    bitxor, BitXorAssign, mixed_bitxor;

    shl, ShlAssign, mixed_shl;
    shr, ShrAssign, mixed_shr;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ops() {
        let mut x = [1.0_f32; 15];
        let y = [2.0_f32; 15];
        let z = [3.0_f32; 15];

        x.add::<8>(&y);

        assert_eq!(x, z);
    }
}

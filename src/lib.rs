#![feature(const_trait_impl)]
#![feature(platform_intrinsics)]
#![feature(portable_simd)]
#![feature(repr_simd)]
#![feature(slice_as_chunks)]
#![no_std]

use core::{hint, iter, mem, ops, simd::SimdElement};

mod intrinsics;

#[derive(Clone, Copy, Debug)]
#[repr(simd)]
struct Simd<T, const N: usize>([T; N]);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Lanes {
    Zero,
    One,
    Two,
    Four,
    Eight,
    Sixteen,
    ThirtyTwo,
    SixtyFour,
}

impl<T: Copy, const N: usize> Simd<T, N> {
    pub const fn from_array(array: [T; N]) -> Self {
        Self(array)
    }

    pub const fn to_array(self) -> [T; N] {
        self.0
    }
}

impl Lanes {
    /// Returns the nearest supported lane count.
    pub const fn nearest<T>(len: usize) -> Self {
        let size = mem::size_of::<T>();

        // this will be optimized out as `size` is a constant
        assert!(size <= 64, "up to 512-byte vectors are supported");

        // clamp to the maximum supported vector size
        let len = len.min(64 / size);

        // check if it's already a valid lane size
        let lanes = if len.is_power_of_two() {
            len
        } else {
            len.next_power_of_two() >> 1
        };

        // im sure rustc can optimize this
        match lanes {
            0 => Self::Zero,
            1 => Self::One,
            2 => Self::Two,
            4 => Self::Four,
            8 => Self::Eight,
            16 => Self::Sixteen,
            32 => Self::ThirtyTwo,
            64 => Self::SixtyFour,
            // SAFETY: this cannot go above 64
            _ => unsafe { hint::unreachable_unchecked() },
        }
    }
}

macro_rules! slice_ext {
    ($($fn:ident, $trait:ident, $intrinsic:ident, $op:tt;)*) => {
        /// Extension trait for slices of scalars.
        pub trait SliceExt<T: SimdElement> {
            $(
                fn $fn(&mut self, other: &[T])
                where
                    T: ops::$trait;
            )*
        }

        impl<T: SimdElement> SliceExt<T> for [T] {
            $(
                #[inline]
                fn $fn(&mut self, other: &[T])
                where
                    T: ops::$trait,
                {
                    #[inline]
                    fn $fn<T, const LANES: usize>(this: &mut [T], other: &[T])
                    where
                        T: SimdElement + ops::$trait,
                    {
                        let (this_chunks, this_remainder) = this.as_chunks_mut::<LANES>();
                        let (other_chunks, other_remainder) = other.as_chunks::<LANES>();

                        for (x, y) in iter::zip(this_chunks.iter_mut(), other_chunks.iter()) {
                            *x = unsafe {
                                intrinsics::$intrinsic(
                                    Simd::from_array(*x),
                                    Simd::from_array(*y),
                                ).to_array()
                            };
                        }

                        for (x, y) in iter::zip(this_remainder.iter_mut(), other_remainder.iter()) {
                            *x $op *y;
                        }
                    }

                    assert_eq!(self.len(), other.len(), "slices must have the same length");

                    let lanes = Lanes::nearest::<T>(self.len());

                    match lanes {
                        Lanes::Zero => {}
                        Lanes::One => $fn::<T, 1>(self, other),
                        Lanes::Two => $fn::<T, 2>(self, other),
                        Lanes::Four => $fn::<T, 4>(self, other),
                        Lanes::Eight => $fn::<T, 8>(self, other),
                        Lanes::Sixteen => $fn::<T, 16>(self, other),
                        Lanes::ThirtyTwo => $fn::<T, 32>(self, other),
                        Lanes::SixtyFour => $fn::<T, 64>(self, other),
                    };
                }
            )*
        }
    }
}

slice_ext! {
    add, AddAssign, simd_add, +=;
    div, DivAssign, simd_div, /=;
    mul, MulAssign, simd_mul, *=;
    rem, RemAssign, simd_rem, %=;
    sub, SubAssign, simd_sub, -=;

    bitand, BitAndAssign, simd_and, &=;
    bitor, BitOrAssign, simd_or, |=;
    bitxor, BitXorAssign, simd_xor, ^=;

    shl, ShlAssign, simd_shl, <<=;
    shr, ShrAssign, simd_shr, >>=;
}

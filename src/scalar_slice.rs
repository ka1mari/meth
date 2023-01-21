use core::{iter, ops};

macro_rules! scalar_slice_ext {
    ($($fn:ident, $trait:ident, $op:tt;)*) => {
        /// Extension trait for slices of scalars.
        pub trait ScalarSliceExt<T>
        where
            T: Copy,
        {
            $(
                fn $fn(&mut self, other: &[T])
                where
                    T: ops::$trait;
            )*
        }

        impl<T> ScalarSliceExt<T> for [T]
        where
            T: Copy,
        {
            $(
                #[inline]
                fn $fn(&mut self, other: &[T])
                where
                    T: ops::$trait,
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

scalar_slice_ext! {
    scalar_add, AddAssign, +=;
    scalar_div, DivAssign, /=;
    scalar_mul, MulAssign, *=;
    scalar_rem, RemAssign, %=;
    scalar_sub, SubAssign, -=;

    scalar_bitand, BitAndAssign, &=;
    scalar_bitor, BitOrAssign, |=;
    scalar_bitxor, BitXorAssign, ^=;

    scalar_shl, ShlAssign, <<=;
    scalar_shr, ShrAssign, >>=;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_ops() {
        let mut x = [1.0_f32; 16];
        let y = [2.0_f32; 16];
        let z = [3.0_f32; 16];

        x.scalar_add(&y);

        assert_eq!(x, z);
    }
}

use std::path::PathBuf;

/// Convert binary data from kinect-to-x11 to PNGs
#[derive(argh::FromArgs)]
struct Args {
	/// binary data input
	#[argh(positional)]
	in_file: PathBuf,

	/// PNG file output
	#[argh(positional)]
	out_file: PathBuf,
}

#[derive(Debug, Clone, Copy)]
enum Format {
	Float,
	Bgrx,
	Rgbx,
}

impl Format {
	fn from_raw(raw: u32) -> Result<Self, u32> {
		Ok(match raw {
			2 => Self::Float,
			3 => Self::Bgrx,
			4 => Self::Rgbx,
			_ => return Err(raw),
		})
	}
}

fn main() {
	let args: Args = argh::from_env();

	let in_data = std::fs::read(&args.in_file).unwrap();

	let width: u32 = usize::from_ne_bytes(in_data[0..8].try_into().unwrap())
		.try_into()
		.unwrap();
	let height: u32 = usize::from_ne_bytes(in_data[8..16].try_into().unwrap())
		.try_into()
		.unwrap();
	let bytes_per_pixel: u32 = usize::from_ne_bytes(in_data[16..24].try_into().unwrap())
		.try_into()
		.unwrap();
	let format = Format::from_raw(
		u32::try_from(usize::from_ne_bytes(in_data[24..32].try_into().unwrap())).unwrap(),
	)
	.unwrap();
	dbg!(format);
	let data = &in_data[32..];

	let image = image::ImageBuffer::from_fn(width, height, |x, y| {
		let idx = usize::try_from((y * width + x) * bytes_per_pixel).unwrap();
		let pixel = &data[idx..][..usize::try_from(bytes_per_pixel).unwrap()];
		match format {
			Format::Float => {
				let depth = f32::from_ne_bytes(pixel.try_into().unwrap());
				let proportion = depth / 4000.0;
				let value = (proportion * 255.0) as u8;
				image::Rgb([value; 3])
			}
			Format::Rgbx => image::Rgb([pixel[0], pixel[1], pixel[2]]),
			Format::Bgrx => image::Rgb([pixel[2], pixel[1], pixel[1]]),
		}
	});

	image.save(&args.out_file).unwrap();
}

//! The same sparse forward as fastnet.h (transposed weights from the C side), for the bake-off.
#[no_mangle]
pub unsafe extern "C" fn rs_logits(w1t: *const f32, w2t: *const f32, prm: *const f32, nh: i32, nf: i32,
                                   x: *const f32, ok: *const i32, na: i32, lg: *mut f32) {
    let (nh, nf, na) = (nh as usize, nf as usize, na as usize);
    let prm = std::slice::from_raw_parts(prm, nh * nf + nh + nh * nh + nh + na * nh + na);
    let w1t = std::slice::from_raw_parts(w1t, nf * nh);
    let w2t = std::slice::from_raw_parts(w2t, nh * nh);
    let x = std::slice::from_raw_parts(x, nf);
    let ok = std::slice::from_raw_parts(ok, na);
    let lg = std::slice::from_raw_parts_mut(lg, na);
    let b1 = &prm[nh * nf..nh * nf + nh];
    let o2 = nh * nf + nh + nh * nh;
    let b2 = &prm[o2..o2 + nh];
    let wp = &prm[o2 + nh..o2 + nh + na * nh];
    let bp = &prm[o2 + nh + na * nh..o2 + nh + na * nh + na];
    let mut h1 = [0f32; 1024];
    let mut h2 = [0f32; 1024];
    let (h1, h2) = (&mut h1[..nh], &mut h2[..nh]);
    h1.copy_from_slice(b1);
    for (i, &xi) in x.iter().enumerate() {
        if xi == 0.0 { continue; }
        for (h, &w) in h1.iter_mut().zip(&w1t[i * nh..(i + 1) * nh]) { *h += xi * w; }
    }
    h2.copy_from_slice(b2);
    for i in 0..nh {
        let v = h1[i];
        if v <= 0.0 { continue; }
        for (h, &w) in h2.iter_mut().zip(&w2t[i * nh..(i + 1) * nh]) { *h += v * w; }
    }
    for h in h2.iter_mut() { if *h < 0.0 { *h = 0.0; } }
    for a in 0..na {
        if ok[a] == 0 { continue; }
        let r = &wp[a * nh..(a + 1) * nh];
        let mut s = bp[a];
        for (&w, &h) in r.iter().zip(h2.iter()) { s += w * h; }
        lg[a] = s;
    }
}

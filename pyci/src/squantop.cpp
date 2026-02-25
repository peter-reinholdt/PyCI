/* This file is part of PyCI.
 *
 * PyCI is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * PyCI is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PyCI. If not, see <http://www.gnu.org/licenses/>. */

#include <iomanip>
#include <iostream>
#include <regex>

#include <pyci.h>

namespace pyci {

SQuantOp::SQuantOp(void) {
}

SQuantOp::SQuantOp(const SQuantOp &ham)
    : nbasis(ham.nbasis), ecore(ham.ecore), one_mo(ham.one_mo), two_mo(ham.two_mo), h(ham.h),
      v(ham.v), w(ham.w), one_mo_array(ham.one_mo_array), two_mo_array(ham.two_mo_array),
      h_array(ham.h_array), v_array(ham.v_array), w_array(ham.w_array) {
}

SQuantOp::SQuantOp(SQuantOp &&ham) noexcept
    : nbasis(std::exchange(ham.nbasis, 0)), ecore(std::exchange(ham.ecore, 0.0)),
      one_mo(std::exchange(ham.one_mo, nullptr)), two_mo(std::exchange(ham.two_mo, nullptr)),
      h(std::exchange(ham.h, nullptr)), v(std::exchange(ham.v, nullptr)),
      w(std::exchange(ham.w, nullptr)), one_mo_array(std::move(ham.one_mo_array)),
      two_mo_array(std::move(ham.two_mo_array)), h_array(std::move(ham.h_array)),
      v_array(std::move(ham.v_array)), w_array(std::move(ham.w_array)) {
}

namespace {

template<typename T>
T read_parameter(const std::string &header, const std::string &name,
                 const std::string &regex_string) {
    std::regex r(name + regex_string);
    std::smatch m;
    T parameter;
    if (std::regex_search(header, m, r)) {
        std::string parameter_string = m[1];
        if (std::is_same<T, bool>::value) {
            std::transform(parameter_string.begin(), parameter_string.end(),
                           parameter_string.begin(), ::tolower);
            std::istringstream(parameter_string) >> std::boolalpha >> parameter;
        } else {
            std::istringstream(parameter_string) >> parameter;
        }
    } else {
        throw std::invalid_argument(name + " is not found.");
    }
    return parameter;
}

template<typename T>
T read_parameter(const std::string &header, const std::string &name,
                 const std::string &regex_string, T default_value) {
    T parameter;
    try {
        parameter = read_parameter<T>(header, name, regex_string);
    } catch (const std::exception &e) {
        parameter = default_value;
    }
    return parameter;
}

} // namespace

SQuantOp::SQuantOp(const std::string &filename) {
    std::ifstream f(filename);
    if (f.fail())
        throw std::ios_base::failure("Failed to read the FCIDUMP file " + filename);

    std::string header, line;
    while (std::getline(f, line) && line.find("&END") == std::string::npos &&
           line.find("/") == std::string::npos) {
        header += " ";
        header += line;
    }
    header += " &END";

    if (f.eof())
        throw std::ios_base::failure("FCIDUMP has the wrong header");

    /* FCIDUMP regexes copied from https://github.com/quan-tum/CDFCI/. */
    const std::string int_regex = R"([ ]*=[ ]*(\d+))";
    const std::string bool_regex = R"([ ]*=[ .]*(FALSE|TRUE))";

    long norb = read_parameter<int>(header, "NORB", int_regex);
    /* long nelec = read_parameter<int>(header, "NELEC", int_regex); */
    /* long ms2 = read_parameter<int>(header, "MS2", int_regex); */
    bool uhf = read_parameter<bool>(header, "UHF", bool_regex, false);

    nbasis = norb;
    one_mo_array = Array<double>({nbasis, nbasis});
    two_mo_array = Array<double>({nbasis, nbasis, nbasis, nbasis});
    h_array = Array<double>(nbasis);
    v_array = Array<double>({nbasis, nbasis});
    w_array = Array<double>({nbasis, nbasis});
    one_mo = reinterpret_cast<double *>(one_mo_array.request().ptr);
    two_mo = reinterpret_cast<double *>(two_mo_array.request().ptr);
    h = reinterpret_cast<double *>(h_array.request().ptr);
    v = reinterpret_cast<double *>(v_array.request().ptr);
    w = reinterpret_cast<double *>(w_array.request().ptr);

    long n1, n2, n3;
    n1 = nbasis;
    n2 = n1 * n1;
    n3 = n2 * n1;
    ecore = 0;
    std::fill(one_mo, one_mo + n2, static_cast<double>(0.));
    std::fill(two_mo, two_mo + n3 * n1, static_cast<double>(0.));
    if (uhf) {
        throw std::runtime_error("Unrestricted FCIDUMP not implemented");
    } else {
        long i, j, k, l;
        double integral;
        while (f >> integral >> i >> j >> k >> l) {
            if (i && j && k && l) {
                --i;
                --j;
                --k;
                --l;
                two_mo[i * n3 + k * n2 + j * n1 + l] = integral;
                two_mo[k * n3 + i * n2 + l * n1 + j] = integral;
                two_mo[j * n3 + k * n2 + i * n1 + l] = integral;
                two_mo[i * n3 + l * n2 + j * n1 + k] = integral;
                two_mo[j * n3 + l * n2 + i * n1 + k] = integral;
                two_mo[l * n3 + j * n2 + k * n1 + i] = integral;
                two_mo[k * n3 + j * n2 + l * n1 + i] = integral;
                two_mo[l * n3 + i * n2 + k * n1 + j] = integral;
            } else if (i && j) {
                --i;
                --j;
                one_mo[i * n1 + j] = integral;
                one_mo[j * n1 + i] = integral;
            } else {
                ecore = integral;
            }
        }
    }
    long i, j, k = 0, l = 0;
    for (i = 0; i != n1; ++i) {
        h[k++] = one_mo[i * (n1 + 1)];
        for (j = 0; j != n1; ++j) {
            v[l] = two_mo[i * n3 + i * n2 + j * n1 + j];
            w[l++] =
                two_mo[i * n3 + j * n2 + i * n1 + j] * 2 - two_mo[i * n3 + j * n2 + j * n1 + i];
        }
    }
}

SQuantOp::SQuantOp(const double e, const Array<double> mo1, const Array<double> mo2)
    : nbasis(mo1.request().shape[0]), ecore(e), one_mo_array(mo1), two_mo_array(mo2),
      h_array(nbasis), v_array({nbasis, nbasis}), w_array({nbasis, nbasis}),
      JKscreen_array({nbasis, nbasis}), Jscreen_array({nbasis,nbasis}) {
    one_mo = reinterpret_cast<double *>(one_mo_array.request().ptr);
    two_mo = reinterpret_cast<double *>(two_mo_array.request().ptr);
    h = reinterpret_cast<double *>(h_array.request().ptr);
    v = reinterpret_cast<double *>(v_array.request().ptr);
    w = reinterpret_cast<double *>(w_array.request().ptr);
    // maximum absolute value for (ii,jj) on antisymmetrized/regular 2e integrals
    // (used for screening)
    JKscreen = reinterpret_cast<double *>(JKscreen_array.request().ptr);
    Jscreen = reinterpret_cast<double *>(Jscreen_array.request().ptr);
    long n1 = nbasis;
    long n2 = nbasis * n1;
    long n3 = nbasis * n2;
    long i, j, k = 0, l = 0;
    long ii, jj, ll, kk;
    hmax = 0.0;
    Jmax = 0.0;
    JKmax = 0.0;
    for (i = 0; i != n1; ++i) {
        ii = i;
        h[k++] = one_mo[i * (n1 + 1)];
        for (j = 0; j != n1; ++j) {
            jj = j;
            double JKmax_ij = 0.0;
            double Jmax_ij = 0.0;
            for (kk = 0; kk != n1; ++kk) {
                for (ll = 0; ll != n1; ++ll) {
                    double aval = std::abs(two_mo[n3 * ii + n2 * kk + n1 * jj + ll] - two_mo[n3 * ii + n2 * kk + n1 * ll + jj]);
                    JKmax_ij = std::max(JKmax_ij, aval);
                    Jmax_ij = std::max(Jmax_ij, std::abs(two_mo[n3 * ii + n2 * kk + n1 * jj + ll]));
                }
            }
            JKscreen[l] = JKmax_ij;
            Jscreen[l] = Jmax_ij;
            JKmax = std::max(JKmax_ij, JKmax);
            Jmax = std::max(Jmax_ij, Jmax);
            if (i != j){
                hmax = std::max(hmax, std::abs(one_mo[j * n1 + i]));
            }
            v[l] = two_mo[i * n3 + i * n2 + j * n1 + j];
            w[l++] =
                two_mo[i * n3 + j * n2 + i * n1 + j] * 2 - two_mo[i * n3 + j * n2 + j * n1 + i];
        }
    }
}

void SQuantOp::to_file(const std::string &filename, const long nelec, const long ms2,
                  const double tol) const {
    bool uhf = false;
    long n1, n2, n3;
    n1 = nbasis;
    n2 = n1 * n1;
    n3 = n2 * n1;
    std::ofstream f(filename);
    if (f.fail())
        throw std::ios_base::failure("Failed to open the FCIDUMP file " + filename);

    f << "&FCIDUMP\nNORB=" << nbasis << ",\nNELEC=" << nelec << ",\nMS2=" << ms2
      << ",\nUHF=" << (uhf ? ".TRUE." : ".FALSE.") << ",\nORBSYM=";
    for (long i = 0; i != nbasis; ++i)
        f << "1,";
    f << "\nISYM=1,\n&END\n";
    long i, j, k, l;
    double val;
    for (i = 0; i != nbasis; ++i)
        for (j = 0; j <= i; ++j)
            for (k = 0; k != nbasis; ++k)
                for (l = 0; l <= k; ++l)
                    if ((i * (i + 1)) / 2 + j >= (k * (k + 1)) / 2 + l) {
                        val = two_mo[i * n3 + k * n2 + j * n1 + l];
                        if (std::abs(val) > tol)
                            f << std::setw(28) << std::setprecision(20) << std::scientific
                              << val << ' ' << i + 1 << ' ' << j + 1
                              << ' ' << k + 1 << ' ' << l + 1 << "\n";
                    }
    for (i = 0; i != nbasis; ++i)
        for (j = 0; j <= i; ++j) {
            val = one_mo[i * n1 + j];
            if (std::abs(val) > tol)
                f << std::setw(28) << std::setprecision(20) << std::scientific << val
                  << ' ' << i + 1 << ' ' << j + 1 << " 0 0\n";
        }

    f << std::setw(28) << std::setprecision(20) << std::scientific << ecore << " 0 0 0 0\n";
}

template<class WfnType>
void SQuantOp::perform_one_electron_direct(const WfnType &wfn, const long xsize, const bool triplet, const double *x, double *y) const {
    auto worker = [&](const long start, const long end, std::vector<std::pair<long, double>>& buf){
        AlignedVector<ulong> det(wfn.nword2);
        AlignedVector<long> occs(wfn.nocc);
        AlignedVector<long> virs(wfn.nvir);
        long i, j, ii, jj, sign_up;
        long jdet;
        long n1 = wfn.nbasis;
        double val, diag;
        ulong *det_up = &det[0];
        ulong *det_dn = det_up + wfn.nword;
        long *occs_up = &occs[0];
        long *occs_dn = occs_up + wfn.nocc_up;
        long *virs_up = &virs[0];
        long *virs_dn = virs_up + wfn.nvir_up;
        const double sign_triplet = triplet ? -1.0 : 1.0;
        const double very_small = 1e-12;
        for (long idet = start; idet<end; idet++){
            diag = 0.0;
            const ulong *rdet_up = wfn.det_ptr(idet);
            const ulong *rdet_dn = rdet_up + wfn.nword;
            std::memcpy(det_up, rdet_up, sizeof(ulong) * wfn.nword2);
            fill_occs(wfn.nword, rdet_up, occs_up);
            fill_occs(wfn.nword, rdet_dn, occs_dn);
            fill_virs(wfn.nword, wfn.nbasis, rdet_up, virs_up);
            fill_virs(wfn.nword, wfn.nbasis, rdet_dn, virs_dn);
            // loop over spin-up occupied indices
            for (i = 0; i < wfn.nocc_up; ++i) {
                ii = occs_up[i];
                diag += one_mo[(wfn.nbasis + 1) * ii];
                // loop over spin-up virtual indices
                for (j = 0; j < wfn.nvir_up; ++j) {
                    jj = virs_up[j];
                    val = one_mo[n1 * ii + jj];
                    if (std::abs(val) < very_small) {
                        continue;
                    }
                    // 1-0 excitation elements
                    excite_det(ii, jj, det_up);
                    sign_up = phase_single_det(wfn.nword, ii, jj, rdet_up);
                    jdet = wfn.index_det(det_up);
                    if (jdet != -1) {
                        val *= sign_up * x[idet];
                        buf.emplace_back(jdet, val);
                    }
                    excite_det(jj, ii, det_up);
                }
            }
            // loop over spin-down occupied indices
            for (i = 0; i < wfn.nocc_dn; ++i) {
                ii = occs_dn[i];
                diag += sign_triplet * one_mo[(wfn.nbasis + 1) * ii];
                // loop over spin-down virtual indices
                for (j = 0; j < wfn.nvir_dn; ++j) {
                    jj = virs_dn[j];
                    val = one_mo[n1 * ii + jj];
                    if (std::abs(val) < very_small) {
                        continue;
                    }
                    // 0-1 excitation elements
                    excite_det(ii, jj, det_dn);
                    jdet = wfn.index_det(det_up);
                    if (jdet != -1) {
                        val *= sign_triplet * phase_single_det(wfn.nword, ii, jj, rdet_dn) * x[idet];
                        buf.emplace_back(jdet, val);
                    }
                    excite_det(jj, ii, det_dn);
                }
            }
            buf.emplace_back(idet, diag*x[idet]);
        }
    };
    long nthread = get_num_threads();
    if (nthread > xsize) nthread = xsize;
    Vector<std::thread> v_threads;
    v_threads.reserve(nthread);

    std::atomic<long> next_chunk(0);
    long num_chunks = 1 + xsize / PYCI_CHUNKSIZE_MIN;
    std::fill(y, y + wfn.ndet, 0.0);

    for (long i = 0; i < nthread; ++i) {
        v_threads.emplace_back([&](){
            std::vector<std::pair<long,double>> buf;
            while (true){
                long ichunk = next_chunk.fetch_add(1);
                if (ichunk >= num_chunks) {
                    break;
                }
                long start = end_chunk_idx(ichunk, num_chunks, xsize);
                long end = end_chunk_idx(ichunk + 1, num_chunks, xsize);
                end = std::min(end, xsize);
                buf.clear();
                worker(start, end, buf);
                for (auto &[j,v] : buf) std::atomic_ref<double>(y[j]).fetch_add(v, std::memory_order_relaxed);
                }
            }
        );
    }
    for (auto &thread : v_threads) thread.join();
}

template<class WfnType>
Array<double> SQuantOp::py_one_electron_direct(const WfnType &wfn, const Array<double> x, const bool triplet) const {
    Array<double> y(wfn.ndet);
    auto x_info = x.request();
    long xsize = x_info.size;
    perform_one_electron_direct(wfn, xsize, triplet, reinterpret_cast<const double *>(x_info.ptr), reinterpret_cast<double *>(y.request().ptr));
    return y;
}

template Array<double> SQuantOp::py_one_electron_direct(const FullCIWfn &wfn, const Array<double> x, const bool triplet) const ;



} // namespace pyci

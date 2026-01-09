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

#include <pyci.h>
#include <chrono>
#include <iostream>

namespace pyci {

namespace {

template<class T>
inline void append(AlignedVector<T> &v, const T &t) {
    if (v.size() + 1 >= v.capacity())
        v.reserve(std::lround(PYCI_SPARSEOP_RESIZE_FACTOR * v.size() + 0.5));
    v.push_back(t);
}

} // namespace

SparseOp::SparseOp(const SparseOp &op)
    : nrow(op.nrow), ncol(op.ncol), size(op.size), ecore(op.ecore), symmetric(op.symmetric),
      shape(op.shape), data(op.data), indices(op.indices), indptr(op.indptr) {
}

SparseOp::SparseOp(SparseOp &&op) noexcept
    : nrow(std::exchange(op.nrow, 0)), ncol(std::exchange(op.ncol, 0)),
      size(std::exchange(op.size, 0)), ecore(std::exchange(op.ecore, 0.0)),
      symmetric(std::exchange(op.symmetric, 0)), shape(std::move(op.shape)),
      data(std::move(op.data)), indices(std::move(op.indices)), indptr(std::move(op.indptr)) {
}

SparseOp::SparseOp(const long rows, const long cols, const bool symm)
    : nrow(rows), ncol(cols), size(0), ecore(0.0), symmetric(symm) {
    shape = pybind11::make_tuple(pybind11::cast(nrow), pybind11::cast(ncol));
    append<long>(indptr, 0);
}

SparseOp::SparseOp(const SQuantOp &ham, const DOCIWfn &wfn, const long rows, const long cols,
                   const bool symm)
    : nrow((rows > -1) ? rows : wfn.ndet), ncol((cols > -1) ? cols : wfn.ndet), size(0),
      ecore(ham.ecore), symmetric(symm) {
    append<long>(indptr, 0);
    update<DOCIWfn>(ham, wfn, nrow, ncol, 0);
}

SparseOp::SparseOp(const SQuantOp &ham, const FullCIWfn &wfn, const long rows, const long cols,
                   const bool symm)
    : nrow((rows > -1) ? rows : wfn.ndet), ncol((cols > -1) ? cols : wfn.ndet), size(0),
      ecore(ham.ecore), symmetric(symm) {
    append<long>(indptr, 0);
    update<FullCIWfn>(ham, wfn, nrow, ncol, 0);
}

SparseOp::SparseOp(const SQuantOp &ham, const GenCIWfn &wfn, const long rows, const long cols,
                   const bool symm)
    : nrow((rows > -1) ? rows : wfn.ndet), ncol((cols > -1) ? cols : wfn.ndet), size(0),
      ecore(ham.ecore), symmetric(symm) {
    append<long>(indptr, 0);
    update<GenCIWfn>(ham, wfn, nrow, ncol, 0);
}

pybind11::object SparseOp::dtype(void) const {
    return pybind11::dtype::of<double>();
}

const double *SparseOp::data_ptr(const long index) const {
    return &data[index];
}

const long *SparseOp::indices_ptr(const long index) const {
    return &indices[index];
}

const long *SparseOp::indptr_ptr(const long index) const {
    return &indptr[index];
}

double SparseOp::get_element(const long i, const long j) const {
    const long *start = &indices[indptr[i]];
    const long *end = &indices[indptr[i + 1]];
    const long *e = std::lower_bound(start, end, j);
    return (*e == j) ? data[indptr[i] + e - start] : 0.0;
}

void SparseOp::perform_op(const double *x, double *y) const {
    if (symmetric){
#ifdef HAS_MKL
        return perform_op_symm_mkl(x, y);
#endif
        return perform_op_symm(x, y);
    }
    typedef Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor, long>> SparseMatrix;
    SparseMatrix mat(nrow, ncol, size, &indptr[0], &indices[0], &data[0], 0);
    Eigen::Map<const Eigen::VectorXd> xvec(x, ncol);
    Eigen::Map<Eigen::VectorXd> yvec(y, nrow);
    yvec = mat * xvec;
}

void SparseOp::perform_op_symm(const double *x, double *y) const {
    typedef Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor, long>> SparseMatrix;
    SparseMatrix mat(nrow, ncol, size, &indptr[0], &indices[0], &data[0], 0);
    Eigen::Map<const Eigen::VectorXd> xvec(x, ncol);
    Eigen::Map<Eigen::VectorXd> yvec(y, nrow);
    yvec = mat.selfadjointView<Eigen::Lower>() * xvec;
}

#ifdef HAS_MKL
void SparseOp::perform_op_symm_mkl(const double *x, double *y) const {
    // the sparse matrix is symmetric, stored as lower diagonal
    // this gets the full matvec by interpreting the sparse matrix is a general matrix
    // then doing A*x + A.T*x - diag(A)*x
    // note that this could be done in MKL with
    // the descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC, 
    // but that scales poorly in the number of threads.
    // the implementation has a nthread*nrow memory overhead, 
    // which should be fine when nthread*nrow<<nnz  
    long long* indptr_mkl = const_cast<long long*>(reinterpret_cast<const long long*>(&indptr[0]));
    long long* indices_mkl = const_cast<long long*>(reinterpret_cast<const long long*>(&indices[0]));

    long nthread = get_num_threads();
    long total_nnz = indptr.back();
    long max_threads = std::max(1L, (long)(total_nnz / 1e6)); // each thread should work on at least a million nnz
    nthread = std::min(nthread, max_threads);

    // y <- A[start:end, :].T * x[start:end]  + A[start:end, :] * x
    auto worker = [&](long row_start, long row_end, std::vector<double>*& y_local) {
        mkl_set_num_threads_local(1);
        y_local = new std::vector<double>(nrow, 0.0);
        long nrow_thread = row_end - row_start;
        sparse_matrix_t mkl_handle;
        sparse_status_t status = mkl_sparse_d_create_csr(
            &mkl_handle,
            SPARSE_INDEX_BASE_ZERO,
            nrow_thread, ncol,
            &indptr_mkl[row_start],
            &indptr_mkl[row_start + 1],
            &indices_mkl[0],
            const_cast<double*>(data.data())
        );

        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        descr.diag = SPARSE_DIAG_NON_UNIT;

        if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("MKL CSR create failed");
        status = mkl_sparse_optimize(mkl_handle);
        if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("MKL optimize failed");


        // y[:]<-A[start:end, :].T @ x[start:end]
        status = mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, mkl_handle, descr, x+row_start, 0.0, y_local->data());
        if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("MKL SpMV failed");
        // y[start:end] += A[start:end, :] @ x
        status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mkl_handle, descr, x, 1.0, &(*y_local)[row_start]);
        if (status != SPARSE_STATUS_SUCCESS) throw std::runtime_error("MKL SpMV failed");
        mkl_sparse_destroy(mkl_handle);
    };
    Vector<std::thread> v_threads;
    std::vector<std::vector<double>*> results(nthread, nullptr);
    
//    long total_nnz = indptr.back();
    long target_nnz = total_nnz / nthread;
    long nnz;
    long row_start = 0;
    long row_end = 0;
    for (long i = 0; i < nthread; i++){
        nnz = indptr[row_end] - indptr[row_start];
        while (nnz < target_nnz && row_end < nrow){
            row_end++;
            nnz = indptr[row_end] - indptr[row_start];
        }
        v_threads.emplace_back([&, i, row_start, row_end](){
            worker(row_start, row_end, results[i]);
        });
        row_start = row_end;
    }
    for (auto &thread : v_threads) thread.join();
    std::fill(y, y+nrow, 0.0);
    for (auto* result : results){
        cblas_daxpy(nrow, 1.0, result->data(), 1, y, 1);
        delete result;
    }
    //remove double-counted diagonal
    std::vector<double> diag_times_x(nrow);
    vdMul(nrow, diagonal.data(), x, diag_times_x.data());
    cblas_daxpy(nrow, -1.0, diag_times_x.data(), 1, y, 1);
}
#endif

template<class WfnType>
void SparseOp::perform_Vop_direct(const SQuantOp &ham, const WfnType &wfn, const long Nint, const double eps, const double *x, double *y) const {
    // perform direct hamiltonian-vector multiply (with threshold epsilon)
    long nthread = get_num_threads();
    long chunksize = wfn.ndet / nthread + static_cast<bool>(wfn.ndet % nthread);
    while (nthread > 1 && chunksize < PYCI_CHUNKSIZE_MIN) {
        nthread /= 2;
        chunksize = wfn.ndet / nthread + static_cast<bool>(wfn.ndet % nthread);
    }

    auto worker = [&](const SQuantOp &ham, const FullCIWfn &wfn, const long start, const long end, const long Nint, const double eps, const double *x, std::vector<double>*& y){
        y = new std::vector<double>(wfn.ndet, 0.0);
        AlignedVector<ulong> det(wfn.nword2);
        AlignedVector<long> occs(wfn.nocc);
        AlignedVector<long> virs(wfn.nvir);
        long i, j, k, l, ii, jj, kk, ll, ioffset, koffset, sign_up;
        long jdet;
        long n1 = wfn.nbasis;
        long n2 = n1 * n1;
        long n3 = n1 * n2;
        double val;
        double eps_i;
        bool is_internal;
        ulong *det_up = &det[0];
        ulong *det_dn = det_up + wfn.nword;
        long *occs_up = &occs[0];
        long *occs_dn = occs_up + wfn.nocc_up;
        long *virs_up = &virs[0];
        long *virs_dn = virs_up + wfn.nvir_up;
        for (long idet = start; idet<end; idet++){
            eps_i = eps / std::abs(x[idet]);
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
                ioffset = n3 * ii;
                // loop over spin-up virtual indices
                for (j = 0; j < wfn.nvir_up; ++j) {
                    jj = virs_up[j];
                    // 1-0 excitation elements
                    excite_det(ii, jj, det_up);
                    sign_up = phase_single_det(wfn.nword, ii, jj, rdet_up);
                    val = ham.one_mo[n1 * ii + jj];
                    for (k = 0; k < wfn.nocc_up; ++k) {
                        kk = occs_up[k];
                        koffset = ioffset + n2 * kk;
                        val += ham.two_mo[koffset + n1 * jj + kk] - ham.two_mo[koffset + n1 * kk + jj];
                    }
                    for (k = 0; k < wfn.nocc_dn; ++k) {
                        kk = occs_dn[k];
                        val += ham.two_mo[ioffset + n2 * kk + n1 * jj + kk];
                    }
                    // add contribution if |H*c| > eps
                    if (std::abs(val) > eps_i) {
                        jdet = wfn.index_det(det_up);
                        is_internal = (idet < Nint) && (jdet < Nint);
                        if ((jdet != -1) && !is_internal) {
                            val *= sign_up * x[idet];
                            y->data()[jdet] += val;
                        }
                    }
                    // loop over spin-down occupied indices
                    if (ham.Jscreen[ii * n1 + jj] > eps_i) {
                        for (k = 0; k < wfn.nocc_dn; ++k) {
                            kk = occs_dn[k];
                            koffset = ioffset + n2 * kk;
                            // loop over spin-down virtual indices
                            for (l = 0; l < wfn.nvir_dn; ++l) {
                                ll = virs_dn[l];
                                // 1-1 excitation elements
                                val = ham.two_mo[koffset + n1 * jj + ll];
                                if (std::abs(val) > eps_i) {
                                    excite_det(kk, ll, det_dn);
                                    // add contribution if |H*c| > eps
                                    jdet = wfn.index_det(det_up);
                                    is_internal = (idet < Nint) && (jdet < Nint);
                                    if ((jdet != -1) && !is_internal) {
                                        val *= sign_up * phase_single_det(wfn.nword, kk, ll, rdet_dn) * x[idet];
                                        y->data()[jdet] += val;
                                    }
                                    excite_det(ll, kk, det_dn);
                                }
                            }
                        }
                    }
                    // loop over spin-up occupied indices
                    if (ham.JKscreen[n1 * ii + jj] > eps_i) {
                        for (k = i + 1; k < wfn.nocc_up; ++k) {
                            kk = occs_up[k];
                            koffset = ioffset + n2 * kk;
                            // loop over spin-up virtual indices
                            for (l = j + 1; l < wfn.nvir_up; ++l) {
                                ll = virs_up[l];
                                // 2-0 excitation elements
                                val = ham.two_mo[koffset + n1 * jj + ll] - ham.two_mo[koffset + n1 * ll + jj];
                                if (std::abs(val) > eps_i) {
                                    excite_det(kk, ll, det_up);
                                    // add contribution if |H*c| > eps
                                    jdet = wfn.index_det(det_up);
                                    is_internal = (idet < Nint) && (jdet < Nint);
                                    if ((jdet != -1) && !is_internal) {
                                        val *= phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_up) * x[idet];
                                        y->data()[jdet] += val;
                                    }
                                    excite_det(ll, kk, det_up);
                                }
                            }
                        }
                    }
                    excite_det(jj, ii, det_up);
                }
            }
            // loop over spin-down occupied indices
            for (i = 0; i < wfn.nocc_dn; ++i) {
                ii = occs_dn[i];
                ioffset = n3 * ii;
                // loop over spin-down virtual indices
                for (j = 0; j < wfn.nvir_dn; ++j) {
                    jj = virs_dn[j];
                    // 0-1 excitation elements
                    excite_det(ii, jj, det_dn);
                    val = ham.one_mo[n1 * ii + jj];
                    for (k = 0; k < wfn.nocc_up; ++k) {
                        kk = occs_up[k];
                        val += ham.two_mo[ioffset + n2 * kk + n1 * jj + kk];
                    }
                    for (k = 0; k < wfn.nocc_dn; ++k) {
                        kk = occs_dn[k];
                        koffset = ioffset + n2 * kk;
                        val += ham.two_mo[koffset + n1 * jj + kk] - ham.two_mo[koffset + n1 * kk + jj];
                    }
                    // add contribution if |H*c| > eps
                    if (std::abs(val) > eps_i) {
                        jdet = wfn.index_det(det_up);
                        is_internal = (idet < Nint) && (jdet < Nint);
                        if ((jdet != -1) && !is_internal) {
                            val *= phase_single_det(wfn.nword, ii, jj, rdet_dn) * x[idet];
                            y->data()[jdet] += val;
                    }
                    }
                    // loop over spin-down occupied indices
                    if (ham.JKscreen[ii * n1 + jj] > eps_i) {
                        for (k = i + 1; k < wfn.nocc_dn; ++k) {
                            kk = occs_dn[k];
                            koffset = ioffset + n2 * kk;
                            // loop over spin-down virtual indices
                            for (l = j + 1; l < wfn.nvir_dn; ++l) {
                                ll = virs_dn[l];
                                // 0-2 excitation elements
                                val = ham.two_mo[koffset + n1 * jj + ll] - ham.two_mo[koffset + n1 * ll + jj];
                                if (std::abs(val) > eps_i) {
                                    excite_det(kk, ll, det_dn);
                                    // add determinant if |H*c| > eps and not already in wfn
                                    jdet = wfn.index_det(det_up);
                                    is_internal = (idet < Nint) && (jdet < Nint);
                                    if ((jdet != -1) && !is_internal) {
                                        val *= phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_dn) * x[idet];
                                        y->data()[jdet] += val;
                                    }
                                    excite_det(ll, kk, det_dn);
                                }
                            }
                        }
                    }
                    excite_det(jj, ii, det_dn);
                }
            }
        }
    };


    Vector<std::thread> v_threads;
    v_threads.reserve(nthread);
    std::vector<std::vector<double>*> results(nthread, nullptr);
    for (long i = 0; i < nthread; ++i) {
        long start = end_chunk_idx(i, nthread, wfn.ndet);
        long end = end_chunk_idx(i + 1, nthread, wfn.ndet);
        end = std::min(end, wfn.ndet);
        v_threads.emplace_back([&, i, start, end](){
                worker(ham, wfn, start, end, Nint, eps, x, results[i]);
            }
        );
    }
    std::fill(y, y+wfn.ndet, 0.0);
    for (auto &thread : v_threads) thread.join();
    for (auto *result : results){
        cblas_daxpy(wfn.ndet, 1.0, result->data(), 1, y, 1);
        delete result;
    }
}

void SparseOp::solve_ci(const long n, const double *coeffs, const long ncv, const long maxiter,
                        const double tol, double *evals, double *evecs) const {
    if ((nrow > 1 && n >= nrow) || (nrow == 1 && n > 1)) {
        throw std::invalid_argument("cannot find >=n eigenpairs for sparse operator with n rows");
    } else if (nrow != ncol) {
        throw pybind11::type_error("Can only solve sparse symmetric matrix operators");
    } else if (nrow == 1) {
        *evals = get_element(0, 0) + ecore;
        *evecs = 1.0;
        return;
    }
    typedef Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor, long>> SparseMatrix;
    SparseMatrix mat(nrow, ncol, size, &indptr[0], &indices[0], &data[0], 0);
    Spectra::SparseSymMatProd<double, Eigen::Lower, Eigen::RowMajor, long> op(mat);
    Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double, Eigen::Lower, Eigen::RowMajor, long>>
        eigs(op, n, (ncv != -1) ? ncv : std::min(nrow, std::max(n * 2 + 1, 20L)));


    if (coeffs == nullptr)
        eigs.init();
    else
        eigs.init(coeffs);
    eigs.compute(Spectra::SortRule::SmallestAlge, (maxiter != -1) ? maxiter : n * nrow * 10, tol);
    if (eigs.info() != Spectra::CompInfo::Successful)
        throw std::runtime_error("did not converge");
    DenseVector<double> eigenvalues(evals, n);
    DenseMatrix<double> eigenvectors(evecs, n, nrow);
    eigenvalues = eigs.eigenvalues();
    for (long i = 0; i < n; ++i)
        evals[i] += ecore;
    // This is needed so that the eigenvectors are in the proper order
    // when passed back to Python as NumPy arrays
    eigenvectors.transpose() = eigs.eigenvectors();

}

Array<double> SparseOp::py_matvec(const Array<double> x) const {
    Array<double> y(nrow);
    perform_op(reinterpret_cast<const double *>(x.request().ptr),
               reinterpret_cast<double *>(y.request().ptr));
    return y;
}

Array<double> SparseOp::py_matvec_out(const Array<double> x, Array<double> y) const {
    perform_op(reinterpret_cast<const double *>(x.request().ptr),
               reinterpret_cast<double *>(y.request().ptr));
    return y;
}

template<class WfnType>
Array<double> SparseOp::py_Vmatvec_direct(const SQuantOp &ham, const WfnType &wfn, const long Nint, const double eps, const Array<double> x) const {
    Array<double> y(wfn.ndet);
    perform_Vop_direct(ham, wfn, Nint, eps, reinterpret_cast<const double *>(x.request().ptr),
               reinterpret_cast<double *>(y.request().ptr));
    return y;
}

template Array<double> SparseOp::py_Vmatvec_direct(const SQuantOp &, const FullCIWfn &, const long Nint, const double eps, const Array<double> x) const ;

pybind11::tuple SparseOp::py_solve_ci(const long n, pybind11::object coeffs, const long ncv,
                                      const long maxiter, const double tol) const {
    Array<double> eigvals(n);
    Array<double> eigvecs({n, nrow});
    const double *cptr =
        coeffs.is(pybind11::none())
            ? nullptr
            : reinterpret_cast<const double *>(coeffs.cast<Array<double>>().request().ptr);
    double *evals = reinterpret_cast<double *>(eigvals.request().ptr);
    double *evecs = reinterpret_cast<double *>(eigvecs.request().ptr);
    solve_ci(n, cptr, ncv, maxiter, tol, evals, evecs);
    return pybind11::make_tuple(eigvals, eigvecs);
}

template<class WfnType>
void SparseOp::py_update(const SQuantOp &ham, const WfnType &wfn) {
    update<WfnType>(ham, wfn, wfn.ndet, wfn.ndet, nrow);
}

template void SparseOp::py_update(const SQuantOp &, const DOCIWfn &);

template void SparseOp::py_update(const SQuantOp &, const FullCIWfn &);

template void SparseOp::py_update(const SQuantOp &, const GenCIWfn &);

template<class WfnType>
void SparseOp::update(const SQuantOp &ham, const WfnType &wfn, const long rows, const long cols,
                      const long startrow) {
    AlignedVector<ulong> det(wfn.nword2);
    AlignedVector<long> occs(wfn.nocc);
    AlignedVector<long> virs(wfn.nvir);
    shape = pybind11::make_tuple(pybind11::cast(rows), pybind11::cast(cols));
    nrow = rows;
    ncol = cols;
    indptr.reserve(nrow + 1);

    long added_rows = nrow - startrow;
    long nthread = get_num_threads();

    if (nthread > added_rows) nthread = added_rows;

    Vector<std::thread> v_threads;
    Vector<SparseOp> v_sparseops;
    v_threads.reserve(nthread);
    v_sparseops.reserve(nthread);

    bool symm = true;
    for (long i = 0; i < nthread; ++i) {
        long start = startrow + end_chunk_idx(i, nthread, added_rows);
        long end = startrow + end_chunk_idx(i + 1, nthread, added_rows);
        v_sparseops.emplace_back(nrow, ncol, symm);
        v_sparseops.back().indptr.reserve(end - start + 1);
        auto &sparseop = v_sparseops.back();
        v_threads.emplace_back([start, end, &sparseop, &ham, &wfn]() {
            AlignedVector<ulong> det_thread(wfn.nword2);
            AlignedVector<long> occs_thread(wfn.nocc);
            AlignedVector<long> virs_thread(wfn.nvir);
            for (long idet = start; idet < end; ++idet) {
                sparseop.add_row(ham, wfn, idet, &det_thread[0], &occs_thread[0], &virs_thread[0]);
            }
            for (long idet = 0; idet < end - start; ++idet) {
                sort_row(idet, sparseop);
            }
        });
    }

    for (auto &thread : v_threads) thread.join();
    for (long i = 0; i < nthread; ++i) {
        long offset = indices.size();
        auto &sparseop = v_sparseops[i];
        indices.insert(indices.end(), sparseop.indices.begin(), sparseop.indices.end());
        data.insert(data.end(), sparseop.data.begin(), sparseop.data.end());
        diagonal.insert(diagonal.end(), sparseop.diagonal.begin(), sparseop.diagonal.end());
        for (size_t i = 1; i < sparseop.indptr.size(); ++i) {
            indptr.push_back(offset + sparseop.indptr[i]);
        }
    }
    size = indices.size();
}

template<class WfnType>
void SparseOp::py_update_diagonal(const SQuantOp &ham, const WfnType &wfn) {
    update_diagonal<WfnType>(ham, wfn, wfn.ndet);
}

template void SparseOp::py_update_diagonal(const SQuantOp &, const FullCIWfn &);

template<class WfnType>
void SparseOp::update_diagonal(const SQuantOp &ham, const WfnType &wfn, const long ndet) {
    long ndet_start = diagonal.size();
    diagonal.resize(ndet);
    long ndet_added = ndet - ndet_start;
    long nthread = get_num_threads();

    if (nthread > ndet_added) nthread = ndet_added;

    Vector<std::thread> v_threads;
    for (long i = 0; i < nthread; ++i) {
        long start = ndet_start + end_chunk_idx(i, nthread, ndet_added);
        long end = ndet_start + end_chunk_idx(i + 1, nthread, ndet_added);
        v_threads.emplace_back([this, start, end, &ham, &wfn]() {
            AlignedVector<ulong> det(wfn.nword2);
            AlignedVector<long> occs(wfn.nocc);
            long *occs_up = &occs[0];
            long *occs_dn = occs_up + wfn.nocc_up;
            long i, j, k, l, ioffset, koffset;
            const long n1 = wfn.nbasis;
            const long n2 = n1 * n1;
            const long n3 = n1 * n2;
            for (long idet = start; idet < end; ++idet){
                double diag = 0.0;
                const ulong *det_up = wfn.det_ptr(idet);
                const ulong *det_dn = det_up + wfn.nword;
                fill_occs(wfn.nword, det_up, occs_up);
                fill_occs(wfn.nword, det_dn, occs_dn);
                for (i = 0; i < wfn.nocc_up; ++i) {
                    j = occs_up[i];
                    ioffset = n3 * j;
                    diag += ham.one_mo[(wfn.nbasis + 1) * j];
                    for (k = i + 1; k < wfn.nocc_up; ++k) {
                        l = occs_up[k];
                        koffset = ioffset + n2 * l;
                        diag += ham.two_mo[koffset + wfn.nbasis * j + l] - ham.two_mo[koffset + wfn.nbasis * l + j];
                    }
                    for (k = 0; k < wfn.nocc_dn; ++k) {
                        l = occs_dn[k];
                        diag += ham.two_mo[ioffset + n2 * l + wfn.nbasis * j + l];
                    }
                }
                for (i = 0; i < wfn.nocc_dn; ++i) {
                    j = occs_dn[i];
                    ioffset = n3 * j;
                    diag += ham.one_mo[(wfn.nbasis + 1) * j];
                    for (k = i + 1; k < wfn.nocc_dn; ++k) {
                        l = occs_dn[k];
                        koffset = ioffset + n2 * l;
                        diag += ham.two_mo[koffset + wfn.nbasis * j + l] - ham.two_mo[koffset + wfn.nbasis * l + j];
                    }
                }
                diagonal[idet] = diag;
            }
        });
    }
    for (auto &thread : v_threads) thread.join();
}

void SparseOp::reserve(const long n) {
    indices.reserve(n);
    data.reserve(n);
}

void SparseOp::squeeze(void) {
    indptr.shrink_to_fit();
    indices.shrink_to_fit();
    data.shrink_to_fit();
}

void SparseOp::sort_row(const long idet, SparseOp &sparseop) {
    typedef std::sort_with_arg::value_iterator_t<double, long> iter;
    long start = sparseop.indptr[idet];
    long end = sparseop.indptr[idet + 1];
    std::sort(iter(&sparseop.data[start], &sparseop.indices[start]), iter(&sparseop.data[end], &sparseop.indices[end]));
}

void SparseOp::add_row(const SQuantOp &ham, const DOCIWfn &wfn, const long idet, ulong *det, long *occs,
                       long *virs) {
    /* long i, j, k, l, jdet, jmin = symmetric ? idet - 1 : -1; */
    long  jdet, jmin = symmetric ? idet : Max<long>();
    double val1 = 0.0, val2 = 0.0;
    wfn.copy_det(idet, det);
    fill_occs(wfn.nword, det, occs);
    fill_virs(wfn.nword, wfn.nbasis, det, virs);
    // loop over occupied indices
    for (long i = 0, j, k, l ; i < wfn.nocc_up; ++i) {
        k = occs[i];
        // compute part of diagonal matrix element
        val1 += ham.v[k * (wfn.nbasis + 1)];
        val2 += ham.h[k];
        for (j = i + 1; j < wfn.nocc_up; ++j)
            val2 += ham.w[k * wfn.nbasis + occs[j]];
        // loop over virtual indices
        for (j = 0; j < wfn.nvir_up; ++j) {
            // compute single/"pair"-excited elements
            l = virs[j];
            excite_det(k, l, det);
            jdet = wfn.index_det(det);
            // check if excited determinant is in wfn
            if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                // add single/"pair"-excited matrix element
                append<double>(data, ham.v[k * wfn.nbasis + l]);
                append<long>(indices, jdet);
            }
            excite_det(l, k, det);
        }
    }
    // add diagonal element to matrix
    if (idet < ncol) {
        append<double>(data, val1 + val2 * 2);
        append<long>(indices, idet);
    }
    // add pointer to next row's indices
    append<long>(indptr, indices.size());
}

void SparseOp::add_row(const SQuantOp &ham, const FullCIWfn &wfn, const long idet, ulong *det_up,
                       long *occs_up, long *virs_up) {
    long i, j, k, l, ii, jj, kk, ll, jdet, jmin = symmetric ? idet : Max<long>();
    long ioffset, koffset, sign_up;
    long n1 = wfn.nbasis;
    long n2 = n1 * n1;
    long n3 = n1 * n2;
    double val1, val2 = 0.0;
    const ulong *rdet_up = wfn.det_ptr(idet);
    const ulong *rdet_dn = rdet_up + wfn.nword;
    ulong *det_dn = det_up + wfn.nword;
    long *occs_dn = occs_up + wfn.nocc_up;
    long *virs_dn = virs_up + wfn.nvir_up;
    std::memcpy(det_up, rdet_up, sizeof(ulong) * wfn.nword2);
    fill_occs(wfn.nword, rdet_up, occs_up);
    fill_occs(wfn.nword, rdet_dn, occs_dn);
    fill_virs(wfn.nword, wfn.nbasis, rdet_up, virs_up);
    fill_virs(wfn.nword, wfn.nbasis, rdet_dn, virs_dn);
    // loop over spin-up occupied indices
    for (i = 0; i < wfn.nocc_up; ++i) {
        ii = occs_up[i];
        ioffset = n3 * ii;
        // compute part of diagonal matrix element
        val2 += ham.one_mo[(n1 + 1) * ii];
        for (k = i + 1; k < wfn.nocc_up; ++k) {
            kk = occs_up[k];
            koffset = ioffset + n2 * kk;
            val2 += ham.two_mo[koffset + n1 * ii + kk] - ham.two_mo[koffset + n1 * kk + ii];
        }
        for (k = 0; k < wfn.nocc_dn; ++k) {
            kk = occs_dn[k];
            val2 += ham.two_mo[ioffset + n2 * kk + n1 * ii + kk];
        }
        // loop over spin-up virtual indices
        for (j = 0; j < wfn.nvir_up; ++j) {
            jj = virs_up[j];
            // 1-0 excitation elements
            excite_det(ii, jj, det_up);
            sign_up = phase_single_det(wfn.nword, ii, jj, rdet_up);
            jdet = wfn.index_det(det_up);
            // check if 1-0 excited determinant is in wfn
            if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                // compute 1-0 matrix element
                val1 = ham.one_mo[n1 * ii + jj];
                for (k = 0; k < wfn.nocc_up; ++k) {
                    kk = occs_up[k];
                    koffset = ioffset + n2 * kk;
                    val1 += ham.two_mo[koffset + n1 * jj + kk] - ham.two_mo[koffset + n1 * kk + jj];
                }
                for (k = 0; k < wfn.nocc_dn; ++k) {
                    kk = occs_dn[k];
                    val1 += ham.two_mo[ioffset + n2 * kk + n1 * jj + kk];
                }
                // add 1-0 matrix element
                if (abs(val1) > 1e-10 ) { 
                    append<double>(data, sign_up * val1);
                    append<long>(indices, jdet);
                }
            }
            // loop over spin-down occupied indices
            for (k = 0; k < wfn.nocc_dn; ++k) {
                kk = occs_dn[k];
                koffset = ioffset + n2 * kk;
                // loop over spin-down virtual indices
                for (l = 0; l < wfn.nvir_dn; ++l) {
                    ll = virs_dn[l];
                    // 1-1 excitation elements
                    excite_det(kk, ll, det_dn);
                    jdet = wfn.index_det(det_up);
                    // check if 1-1 excited determinant is in wfn
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                        // add 1-1 matrix element
                        if (abs(ham.two_mo[koffset + n1 * jj + ll]) > 1e-10) {
                            append<double>(data, sign_up *
                                                     phase_single_det(wfn.nword, kk, ll, rdet_dn) *
                                                     ham.two_mo[koffset + n1 * jj + ll]);
                            append<long>(indices, jdet);
                        }
                    }
                    excite_det(ll, kk, det_dn);
                }
            }
            // loop over spin-up occupied indices
            for (k = i + 1; k < wfn.nocc_up; ++k) {
                kk = occs_up[k];
                koffset = ioffset + n2 * kk;
                // loop over spin-up virtual indices
                for (l = j + 1; l < wfn.nvir_up; ++l) {
                    ll = virs_up[l];
                    // 2-0 excitation elements
                    excite_det(kk, ll, det_up);
                    jdet = wfn.index_det(det_up);
                    // check if 2-0 excited determinant is in wfn
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                        // add 2-0 matrix element
                        if (abs((ham.two_mo[koffset + n1 * jj + ll] - ham.two_mo[koffset + n1 * ll + jj])) > 1e-10) {
                            append<double>(data, phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_up) *
                                                     (ham.two_mo[koffset + n1 * jj + ll] -
                                                      ham.two_mo[koffset + n1 * ll + jj]));
                            append<long>(indices, jdet);
                        }
                    }
                    excite_det(ll, kk, det_up);
                }
            }
            excite_det(jj, ii, det_up);
        }
    }
    // loop over spin-down occupied indices
    for (i = 0; i < wfn.nocc_dn; ++i) {
        ii = occs_dn[i];
        ioffset = n3 * ii;
        // compute part of diagonal matrix element
        val2 += ham.one_mo[(n1 + 1) * ii];
        for (k = i + 1; k < wfn.nocc_dn; ++k) {
            kk = occs_dn[k];
            koffset = ioffset + n2 * kk;
            val2 += ham.two_mo[koffset + n1 * ii + kk] - ham.two_mo[koffset + n1 * kk + ii];
        }
        // loop over spin-down virtual indices
        for (j = 0; j < wfn.nvir_dn; ++j) {
            jj = virs_dn[j];
            // 0-1 excitation elements
            excite_det(ii, jj, det_dn);
            jdet = wfn.index_det(det_up);
            // check if 0-1 excited determinant is in wfn
            if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                // compute 0-1 matrix element
                val1 = ham.one_mo[n1 * ii + jj];
                for (k = 0; k < wfn.nocc_up; ++k) {
                    kk = occs_up[k];
                    val1 += ham.two_mo[ioffset + n2 * kk + n1 * jj + kk];
                }
                for (k = 0; k < wfn.nocc_dn; ++k) {
                    kk = occs_dn[k];
                    koffset = ioffset + n2 * kk;
                    val1 += ham.two_mo[koffset + n1 * jj + kk] - ham.two_mo[koffset + n1 * kk + jj];
                }
                // add 0-1 matrix element
                if (abs(val1) > 1e-10) { 
                    append<double>(data, phase_single_det(wfn.nword, ii, jj, rdet_dn) * val1);
                    append<long>(indices, jdet);
                }
            }
            // loop over spin-down occupied indices
            for (k = i + 1; k < wfn.nocc_dn; ++k) {
                kk = occs_dn[k];
                koffset = ioffset + n2 * kk;
                // loop over spin-down virtual indices
                for (l = j + 1; l < wfn.nvir_dn; ++l) {
                    ll = virs_dn[l];
                    // 0-2 excitation elements
                    excite_det(kk, ll, det_dn);
                    jdet = wfn.index_det(det_up);
                    // check if excited determinant is in wfn
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                        // add 0-2 matrix element
                        if (abs((ham.two_mo[koffset + n1 * jj + ll] - ham.two_mo[koffset + n1 * ll + jj])) > 1e-10) {
                            append<double>(data, phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_dn) *
                                                     (ham.two_mo[koffset + n1 * jj + ll] -
                                                      ham.two_mo[koffset + n1 * ll + jj]));
                            append<long>(indices, jdet);
                        }
                    }
                    excite_det(ll, kk, det_dn);
                }
            }
            excite_det(jj, ii, det_dn);
        }
    }
    // add diagonal element to matrix
    if (idet < ncol) {
        append<double>(data, val2);
        append<long>(indices, idet);
        append<double>(diagonal, val2);
    }
    // add pointer to next row's indices
    append<long>(indptr, indices.size());
}

void SparseOp::add_row(const SQuantOp &ham, const GenCIWfn &wfn, const long idet, ulong *det, long *occs,
                       long *virs) {
    long jdet, jmin = symmetric ? idet : Max<long>();
    long n1 = wfn.nbasis;
    long n2 = n1 * n1;
    long n3 = n1 * n2;
    double val1, val2 = 0.0;
    const ulong *rdet = wfn.det_ptr(idet);
    // fill working vectors
    std::memcpy(det, rdet, sizeof(ulong) * wfn.nword);
    fill_occs(wfn.nword, rdet, occs);
    fill_virs(wfn.nword, wfn.nbasis, rdet, virs);
    // loop over occupied indices
    for (long i = 0, j, k, l, ii, jj, kk, ll, ioffset, koffset; i < wfn.nocc; ++i) {
        ii = occs[i];
        ioffset = n3 * ii;
        // compute part of diagonal matrix element
        val2 += ham.one_mo[(n1 + 1) * ii];
        for (k = i + 1; k < wfn.nocc; ++k) {
            kk = occs[k];
            koffset = ioffset + n2 * kk;
            val2 += ham.two_mo[koffset + n1 * ii + kk] - ham.two_mo[koffset + n1 * kk + ii];
        }
        // loop over virtual indices
        for (j = 0; j < wfn.nvir; ++j) {
            jj = virs[j];
            // single excitation elements
            excite_det(ii, jj, det);
            jdet = wfn.index_det(det);
            // check if singly-excited determinant is in wfn
            if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                // compute single excitation matrix element
                val1 = ham.one_mo[n1 * ii + jj];
                for (k = 0; k < wfn.nocc; ++k) {
                    kk = occs[k];
                    koffset = ioffset + n2 * kk;
                    val1 += ham.two_mo[koffset + n1 * jj + kk] - ham.two_mo[koffset + n1 * kk + jj];
                }
                // add single excitation matrix element
                append<double>(data, phase_single_det(wfn.nword, ii, jj, rdet) * val1);
                append<long>(indices, jdet);
            }
            // loop over occupied indices
            for (k = i + 1; k < wfn.nocc; ++k) {
                kk = occs[k];
                koffset = ioffset + n2 * kk;
                // loop over virtual indices
                for (l = j + 1; l < wfn.nvir; ++l) {
                    ll = virs[l];
                    // double excitation elements
                    excite_det(kk, ll, det);
                    jdet = wfn.index_det(det);
                    // check if double excited determinant is in wfn
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                        // add double matrix element
                        append<double>(data, phase_double_det(wfn.nword, ii, kk, jj, ll, rdet) *
                                                 (ham.two_mo[koffset + n1 * jj + ll] -
                                                  ham.two_mo[koffset + n1 * ll + jj]));
                        append<long>(indices, jdet);
                    }
                    excite_det(ll, kk, det);
                }
            }
            excite_det(jj, ii, det);
        }
    }
    // add diagonal element to matrix
    if (idet < ncol) {
        append<double>(data, val2);
        append<long>(indices, idet);
    }
    // add pointer to next row's indices
    append<long>(indptr, indices.size());
}

pybind11::array_t<double> SparseOp::py_data() const {
    ssize_t size = static_cast<ssize_t>(data.size());
    return pybind11::array_t<double>(
            {size},
            {sizeof(double)},
            data.data(),
            pybind11::cast(this)
    );
}

pybind11::array_t<long> SparseOp::py_indices() const {
    ssize_t size = static_cast<ssize_t>(indices.size());
    return pybind11::array_t<long>(
            {size},
            {sizeof(long)},
            indices.data(),
            pybind11::cast(this)
    );
}

pybind11::array_t<long> SparseOp::py_indptr() const {
    ssize_t size = static_cast<ssize_t>(indptr.size());
    return pybind11::array_t<long>(
            {size},
            {sizeof(long)},
            indptr.data(),
            pybind11::cast(this)
    );
}

pybind11::array_t<double> SparseOp::py_diagonal() const {
    ssize_t size = static_cast<ssize_t>(diagonal.size());
    return pybind11::array_t<double>(
            {size},
            {sizeof(double)},
            diagonal.data(),
            pybind11::cast(this)
    );
}

} // namespace pyci

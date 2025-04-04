! Optimized verison compared to the stanard one as it use D2Z in the first step
! This avoid transpostions with nx x ny x nz complex and instead use nx/2+1 x ny x nz


#define CHECK_CUDECOMP_EXIT(f) if (f /= CUDECOMP_RESULT_SUCCESS) call exit(1)
!#define CHECK_CUDECOMP_EXIT(f) if (f /= CUDECOMP_RESULT_SUCCESS) exit(1)

! Solves poisson equation
!
!   u_xx + u_yy + u_zz= phi(x,y,z)
!
! where
!
!   phi(x,y,z) = sin(Mx*x)*sin(My*y)*sin(Mz*z)
!
! on domain 0 <= x <= 2*pi, 0 <= y <= 2*pi, 0 <= z <= 2*pi

program main
  use cudafor
  use cudecomp
  use cufft
  use mpi

  implicit none

  ! Command line arguments
  ! grid dimensions
  integer :: nx, ny, nz
  integer :: comm_backend
  integer :: pr, pc

  ! grid size
  real(8), parameter :: Lx = 1.0, Ly = 1.0, Lz = 1.0
  ! modes for phi
  integer, parameter :: Mx = 1, My = 2, Mz = 1
  real(8), parameter :: twopi = 8.0_8*atan(1.0_8)
  real(8) :: err, maxErr = -1.0, times, timef
  character(len=40) :: namefile


  ! grid
  real(8), allocatable :: x(:)
  real(8) :: hx, hy, hz
  ! wavenumbers
  real(8), allocatable :: kx(:)
  real(8), device, allocatable :: kx_d(:)

  real(8), allocatable :: phi(:), ua(:,:,:), rhsp(:,:,:), p(:,:,:)
  !complex(8), allocatable :: rhspc(:,:,:)
  complex(8), device, allocatable :: phi_d(:)
  complex(8), pointer, device, contiguous :: work_d(:)

  ! MPI
  integer :: rank, ranks, ierr
  integer :: localRank, localComm

  ! cudecomp
  type(cudecompHandle) :: handle
  type(cudecompGridDesc) :: grid_desc
  type(cudecompGridDescConfig) :: config
  type(cudecompGridDescAutotuneOptions) :: options

  integer :: pdims(2)
  integer :: gdims(3)
  integer :: npx, npy, npz
  type(cudecompPencilInfo) :: piX, piY, piZ
  integer(8) :: nElemX, nElemY, nElemZ, nElemWork

  ! CUFFT
  integer :: planX, planY, planZ, planXF, planXB
  integer :: batchsize
  integer :: status

  logical :: skip_next
  character(len=16) :: arg
  integer :: i, j ,k , jl, kl, jg, kg


  ! MPI initialization

  call mpi_init(ierr)
  if (ierr /= MPI_SUCCESS) write(*,*) 'mpi_init failed: ', ierr
  call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)
  if (ierr /= MPI_SUCCESS) write(*,*) 'mpi_comm_rank failed: ', ierr
  call mpi_comm_size(MPI_COMM_WORLD, ranks, ierr)
  if (ierr /= MPI_SUCCESS) write(*,*) 'mpi_comm_size failed: ', ierr

  call mpi_comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, localComm, ierr)
  if (ierr /= MPI_SUCCESS) write(*,*) 'mpi_comm_split_type failed: ', ierr
  call mpi_comm_rank(localComm, localRank, ierr)
  if (ierr /= MPI_SUCCESS) write(*,*) 'mpi_comm_rank on local rank failed: ', ierr
  ierr = cudaSetDevice(localRank)

  ! Parse command-line arguments
  nx = 64
  ny = nx
  nz = nx
  pr = 2
  pc = 1
  comm_backend = CUDECOMP_TRANSPOSE_COMM_MPI_P2P


  ! cudecomp initialization

  CHECK_CUDECOMP_EXIT(cudecompInit(handle, MPI_COMM_WORLD))

  CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(config))
  pdims = [pr, pc]
  config%pdims = pdims
  gdims = [ nx/2+1, ny, nz]
  config%gdims = gdims
  config%transpose_comm_backend = comm_backend
  config%transpose_axis_contiguous = .true.

  CHECK_CUDECOMP_EXIT(cudecompGridDescAutotuneOptionsSetDefaults(options))
  options%dtype = CUDECOMP_DOUBLE_COMPLEX
  if (comm_backend == 0) then
    options%autotune_transpose_backend = .true.
  endif

  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, grid_desc, config, options))

  if (rank == 0) then
     write(*,"('Running on ', i0, ' x ', i0, ' process grid ...')") config%pdims(1), config%pdims(2)
     write(*,"('Using ', a, ' backend ...')") cudecompTransposeCommBackendToString(config%transpose_comm_backend)
  end if

  ! get pencil info
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, piX, 1))
  nElemX = piX%size
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, piY, 2))
  nElemY = piY%size
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, piZ, 3))
  nElemZ = piZ%size

  ! get workspace size
  CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_desc, nElemWork))

  ! CUFFT initialization

  batchSize = piX%shape(2)*piX%shape(3)
  status = cufftPlan1D(planX, nx, CUFFT_Z2Z, batchSize)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating X plan'

  batchSize = piX%shape(2)*piX%shape(3)
  status = cufftPlan1D(planXF, nx, CUFFT_D2Z, batchSize)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating X-forward plan'

  batchSize = piX%shape(2)*piX%shape(3)
  status = cufftPlan1D(planXB, nx, CUFFT_Z2D, batchSize)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating X-backward plan'

  batchSize = piY%shape(2)*piY%shape(3)
  status = cufftPlan1D(planY, ny, CUFFT_Z2Z, batchSize)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating Y plan'

  batchSize = piZ%shape(2)*piZ%shape(3)
  status = cufftPlan1D(planZ, nz, CUFFT_Z2Z, batchSize)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating Z plan'


  ! Physical grid

  allocate(x(nx))

  hx = twopi/nx
  do i = 1, nx
     x(i) = hx*i
  enddo
  ! Wavenumbers

  allocate(kx(nx))

  do i = 1, nx/2
     kx(i) = (i-1)
  enddo
  do i = nx/2+1, nx
     kx(i) = (i-1-nx)
  enddo
  allocate(kx_d, source=kx)

  ! allocate arrays

  allocate(phi(max(nElemX, nElemY, nElemZ)))
  allocate(phi_d, mold=phi)
  allocate(ua(nx, piX%shape(2), piX%shape(3)))
  allocate(rhsp(nx, piX%shape(2), piX%shape(3)))
  !allocate(rhspc(nx, piX%shape(2), piX%shape(3)))
  allocate(p(nx, piX%shape(2), piX%shape(3)))


  CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_desc, work_d, nElemWork))

 !initialize rigth hand side + analytic solution
    do kl = 1, piX%shape(3)
       kg = piX%lo(3) + kl - 1
       do jl = 1, piX%shape(2)
          jg = piX%lo(2) + jl - 1
          do i = 1, nx
             rhsp(i,jl,kl) = sin(Mx*x(i))*sin(My*x(jg))*sin(Mz*x(kg))
             !rhspc(i,jl,kl) = cmplx(rhsp(i,jl,kl),0.d0)
             ua(i,jl,kl) = -rhsp(i,jl,kl)/(Mx**2 + My**2 + Mz**2)
          enddo
       enddo
  enddo

   write(namefile,'(a,i3.3,a)') 'rhsp_',rank,'.dat'
   open(unit=55,file=namefile,form='unformatted',position='append',access='stream',status='new')
   write(55) rhsp
   close(55)

  call cpu_time(times)

  ! phi(x,y,z) -> phi(kx,y,z)
  !$acc host_data use_device(rhsp)
  status = cufftExecD2Z(planXF, rhsp, phi_d)
  if (status /= CUFFT_SUCCESS) write(*,*) 'X forward error: ', status
  !$acc end host_data
  ! phi(kx,y,z) -> phi(y,z,kx)
  CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(handle, grid_desc, phi_d, phi_d, work_d, CUDECOMP_DOUBLE_COMPLEX))
  ! phi(y,z,kx) -> phi(ky,z,kx)
  status = cufftExecZ2Z(planY, phi_d, phi_d, CUFFT_FORWARD)
  if (status /= CUFFT_SUCCESS) write(*,*) 'Y forward error: ', status
  ! phi(ky,z,kx) -> phi(z,kx,ky)
  CHECK_CUDECOMP_EXIT(cudecompTransposeYToZ(handle, grid_desc, phi_d, phi_d, work_d, CUDECOMP_DOUBLE_COMPLEX))
  ! phi(z,kx,ky) -> phi(kz,kx,ky)
  status = cufftExecZ2Z(planZ, phi_d, phi_d, CUFFT_FORWARD)
  if (status /= CUFFT_SUCCESS) write(*,*) 'Z forward error: ', status


  block
    complex(8), device, pointer :: phi3d(:,:,:)
    real(8) :: k2
    integer :: il, jl, ig, jg
    integer :: offsets(3), xoff, yoff
    integer :: np(3)
    np(piZ%order(1)) = piZ%shape(1)
    np(piZ%order(2)) = piZ%shape(2)
    np(piZ%order(3)) = piZ%shape(3)
    call c_f_pointer(c_devloc(phi_d), phi3d, piZ%shape)


    ! divide by -K**2, and normalize

    offsets(piZ%order(1)) = piZ%lo(1) - 1
    offsets(piZ%order(2)) = piZ%lo(2) - 1
    offsets(piZ%order(3)) = piZ%lo(3) - 1

    xoff = offsets(1)
    yoff = offsets(2)
    npx = np(1)
    npy = np(2)
    !$cuf kernel do (2)
    do jl = 1, npy
       do il = 1, npx
          jg = yoff + jl
          ig = xoff + il
          do k = 1, nz
              k2 = kx_d(ig)**2 + kx_d(jg)**2 + kx_d(k)**2
              !phi3d(k,il,jl) = phi3d(k,il,jl)/(int(nx,8)*int(ny,8)*int(nz,8))    
              phi3d(k,il,jl) = -phi3d(k,il,jl)/k2/(int(nx,8)*int(ny,8)*int(nz,8))
          enddo
       enddo
    enddo

    ! specify mean (corrects division by zero wavenumber above)
    if (xoff == 0 .and. yoff == 0) phi3d(1,1,1) = 0.0

  end block


  ! phi(kz,kx,ky) -> phi(z,kx,ky)
  status = cufftExecZ2Z(planZ, phi_d, phi_d, CUFFT_INVERSE)
  if (status /= CUFFT_SUCCESS) write(*,*) 'Z inverse error: ', status
  ! phi(z,kx,ky) -> phi(ky,z,kx)
  CHECK_CUDECOMP_EXIT(cudecompTransposeZToY(handle, grid_desc, phi_d, phi_d, work_d, CUDECOMP_DOUBLE_COMPLEX))
  ! phi(ky,z,kx) -> phi(y,z,kx)
  status = cufftExecZ2Z(planY, phi_d, phi_d, CUFFT_INVERSE)
  if (status /= CUFFT_SUCCESS) write(*,*) 'Y inverse error: ', status
  ! phi(y,z,kx) -> phi(kx,y,z)
  CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(handle, grid_desc, phi_d, phi_d, work_d, CUDECOMP_DOUBLE_COMPLEX))
  ! phi(kx,y,z) -> phi(x,y,z)
  !$acc host_data use_device(p)
  status = cufftExecZ2D(planXB, phi_d, p)
  if (status /= CUFFT_SUCCESS) write(*,*) 'X inverse error: ', status
  !$acc end host_data

  err=0.d0
  maxErr=-1.d0
  !$acc kernels
    do kl = 1, piX%shape(3)
       do jl = 1, piX%shape(2)
          do i = 1, nx
             err = abs(ua(i,jl,kl)-p(i,jl,kl))
             !rhsp(i,jl,kl)=real(rhspc(i,jl,kl))
             if (err > maxErr) maxErr = err
          enddo
       enddo
    enddo
    !$acc end kernels

    write(namefile,'(a,i3.3,a)') 'rhspo_',rank,'.dat'
    open(unit=55,file=namefile,form='unformatted',position='append',access='stream',status='new')
    write(55) p
    close(55)

    write(namefile,'(a,i3.3,a)') 'ua_',rank,'.dat'
    open(unit=55,file=namefile,form='unformatted',position='append',access='stream',status='new')
    write(55) ua
    close(55)


    write(*,"('[', i0, '] Max Error: ', e12.6)") rank, maxErr

  ! cleanup

  !status = cufftDestroy(planX)
  !if (status /= CUFFT_SUCCESS) write(*,*) 'X plan destroy: ', status
  status = cufftDestroy(planXF)
  if (status /= CUFFT_SUCCESS) write(*,*) 'XF plan destroy: ', status
  status = cufftDestroy(planXB)
  if (status /= CUFFT_SUCCESS) write(*,*) 'XB plan destroy: ', status
  status = cufftDestroy(planY)
  if (status /= CUFFT_SUCCESS) write(*,*) 'Y plan destroy: ', status
  status = cufftDestroy(planZ)
  if (status /= CUFFT_SUCCESS) write(*,*) 'Z plan destroy: ', status

  deallocate(x)
  deallocate(kx)
  deallocate(kx_d)
  deallocate(phi, phi_d, ua)

  CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, work_d))
  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc))
  CHECK_CUDECOMP_EXIT(cudecompFinalize(handle))

  call mpi_finalize(ierr)

end program main

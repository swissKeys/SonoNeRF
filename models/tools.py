import numpy as np



def get_format_dofs(batch_dofs, frames_num, neighbour_slice, cam_cali_mat, merge_option='average_dof'):
            def get_next_pos(trans_params1, dof, cam_cali_mat):
                """
                Given the first frame's Aurora position line and relative 6dof, return second frame's position line
                :param trans_params1: Aurora position line of the first frame
                :param dof: 6 degrees of freedom based on the first frame, rotations should be degrees
                :param cam_cali_mat: Camera calibration matrix of this case
                :return: Aurora position line of the second frame
                """
                trans_mat1 = params_to_mat44(trans_params1, cam_cali_mat=cam_cali_mat)

                """ Transfer degrees to euler """
                dof[3:] = dof[3:] * (2 * math.pi) / 360

                rot_mat = tfms.euler_matrix(dof[5], dof[4], dof[3], 'rzyx')[:3, :3]

                relative_mat = np.identity(4)
                relative_mat[:3, :3] = rot_mat
                relative_mat[:3, 3] = dof[:3]

                next_mat = np.dot(inv(cam_cali_mat), inv(np.dot(relative_mat, trans_mat1)))
                quaternions = tfms.quaternion_from_matrix(next_mat)  # wxyz

                next_params = np.zeros(7)
                next_params[:3] = next_mat[:3, 3]
                next_params[3:6] = quaternions[1:]
                next_params[6] = quaternions[0]
                return next_params
            def get_6dof_label(trans_params1, trans_params2, cam_cali_mat):
                """
                Given two Aurora position lines of two frames, return the relative 6 degrees of freedom label
                Aurora position line gives the transformation from the ultrasound tracker to Aurora
                :param trans_params1: Aurora position line of the first frame
                :param trans_params2: Aurora position line of the second frame
                :param cam_cali_mat: Camera calibration matrix of this case, which is the transformation from
                the ultrasound image upper left corner (in pixel) to the ultrasound tracker (in mm).
                :return: the relative 6 degrees of freedom (3 translations and 3 rotations xyz) as training label
                Note that this dof is based on the position of the first frame
                """
                trans_mat1 = params_to_mat44(trans_params1, cam_cali_mat=cam_cali_mat)
                trans_mat2 = params_to_mat44(trans_params2, cam_cali_mat=cam_cali_mat)

                relative_mat = np.dot(trans_mat2, inv(trans_mat1))

                translations = relative_mat[:3, 3]
                rotations = R.from_matrix(relative_mat[:3, :3])
                rotations_eulers = rotations.as_euler('xyz')

                dof = np.concatenate((translations, rotations_eulers), axis=0)
                return dof
            
            def params_to_mat44(trans_params, cam_cali_mat):
                """
                Transform the parameters in Aurora files into 4 x 4 matrix
                :param trans_params: transformation parameters in Aurora.pos. Only the last 7 are useful
                3 are translations, 4 are the quaternion (x, y, z, w) for rotation
                :return: 4 x 4 transformation matrix
                """
                if trans_params.shape[0] == 9:
                    trans_params = trans_params[2:]

                translation = trans_params[:3]
                quaternion = trans_params[3:]

                """ Transform quaternion to rotation matrix"""
                r_mat = R.from_quat(quaternion).as_matrix()

                trans_mat = np.zeros((4, 4))
                trans_mat[:3, :3] = r_mat
                trans_mat[:3, 3] = translation
                trans_mat[3, 3] = 1

                trans_mat = np.dot(cam_cali_mat, trans_mat)
                trans_mat = inv(trans_mat)

                # new_qua = np.zeros((4, ))
                # new_qua[0] = quaternion[3]
                # new_qua[1:] = quaternion[:3]
                # eulers_from_mat = tfms.euler_from_matrix(r_mat)
                # eulers_from_qua = tfms.euler_from_quaternion(new_qua, axes='sxyz')
                # print('eulers mat\n{}'.format(eulers_from_mat))
                # print('eulers qua\n{}'.format(eulers_from_qua))
                #
                # recon_R = tfms.euler_matrix(eulers_from_mat[0],
                #                             eulers_from_mat[1],
                #                             eulers_from_mat[2])
                # print('R\n{}'.format(r_mat))
                # print('recon_R\n{}'.format(recon_R))
                return trans_mat
            """
            Based on the network outputs, here reformat the result into one row for each frame
            (Because there are many overlapping frames due to the input format)
            :return:
            1) gen_dofs is (slice_num - 1) x 6dof. It is the relative 6dof motion comparing to
            the former frame
            2) pos_params is slice_num x 7params. It is the absolute position, exactly the same
            format as Aurora.pos file
            """
            print('Use <{}> formatting dofs'.format(merge_option))
            if merge_option == 'one':
                gen_dofs = np.zeros((frames_num - 1, 6))
                gen_dofs[:batch_dofs.shape[0], :] = batch_dofs[:, 0, :]
                gen_dofs[batch_dofs.shape[0], :] = batch_dofs[-1, 1, :]
                print('gen_dof shape {}'.format(gen_dofs.shape))
                print('not average method')

            elif merge_option == 'baton':
                print('baton batch_dofs shape {}'.format(batch_dofs.shape))
                print('slice_num {}'.format(frames_num))
                print('neighboring {}'.format(neighbour_slice))

                gen_dofs = []
                slice_params = []
                for slice_idx in range(frames_num):
                    if slice_idx == 0:
                        this_params = self.case_pos[slice_idx, :]
                        slice_params.append(this_params)
                    elif slice_idx < neighbour_slice:
                        this_dof = batch_dofs[0, :] / 4
                        this_params = get_next_pos(trans_params1=slice_params[slice_idx-1],
                                                         dof=this_dof,
                                                         cam_cali_mat=cam_cali_mat)
                        gen_dofs.append(this_dof)
                        slice_params.append(this_params)
                    else:
                        baton_idx = slice_idx - neighbour_slice + 1
                        baton_params = slice_params[baton_idx]
                        sample_dof = batch_dofs[baton_idx, :]
                        this_params = get_next_pos(trans_params1=baton_params,
                                                         dof=sample_dof,
                                                         cam_cali_mat=cam_cali_mat)
                        this_dof = get_6dof_label(trans_params1=slice_params[slice_idx-1],
                                                        trans_params2=this_params,
                                                        cam_cali_mat=cam_cali_mat)
                        gen_dofs.append(this_dof)
                        slice_params.append(this_params)
                gen_dofs = np.asarray(gen_dofs)
                slice_params = np.asarray(slice_params)
                print('gen_dof shape {}'.format(gen_dofs.shape))
                print('slice_params shape {}'.format(slice_params.shape))
                # time.sleep(30)
            else:
                frames_pos = []
                for start_sample_id in range(batch_dofs.shape[0]):
                    for relative_id in range(batch_dofs.shape[1]):
                        this_pos_id = start_sample_id + relative_id + 1
                        # print('this_pos_id {}'.format(this_pos_id))
                        this_pos = batch_dofs[start_sample_id, relative_id, :]
                        this_pos = np.expand_dims(this_pos, axis=0)
                        if len(frames_pos) < this_pos_id:
                            frames_pos.append(this_pos)
                        else:
                            frames_pos[this_pos_id - 1] = np.concatenate((frames_pos[this_pos_id - 1],
                                                                          this_pos), axis=0)

                gen_dofs = []
                for i in range(len(frames_pos)):
                    gen_dof = np.mean(frames_pos[i], axis=0)

                    """This is for Linear Motion"""
                    # gen_dof = train_network.dof_stats[:, 0]
                    # gen_dof = np.asarray([-0.07733258, -1.28508398, 0.37141262,
                    #                       -0.57584312, 0.20969176, 0.51404395]) + 0.1

                    gen_dofs.append(gen_dof)
                gen_dofs = np.asarray(gen_dofs)

                print('batch_dofs {}'.format(batch_dofs.shape))
                print('gen_dofs {}'.format(gen_dofs.shape))
                # time.sleep(30)

            # for dof_id in range(6):
            #     gen_dofs[:, dof_id] = tools.smooth_array(gen_dofs[:, dof_id])
            # time.sleep(30)
            return gen_dofs



def dof2params(format_dofs):
            gen_param_results = []
            for i in range(format_dofs.shape[0]):
                if i == 0:
                    base_param = self.case_pos[i, :]
                else:
                    base_param = gen_param_results[i-1]
                gen_dof = format_dofs[i, :]
                gen_param = tools.get_next_pos(trans_params1=base_param,
                                               dof=gen_dof, cam_cali_mat=self.cam_cali_mat)
                gen_param_results.append(gen_param)
            # time.sleep(30)
            gen_param_results = np.asarray(gen_param_results)
            pos_params = np.zeros((self.frames_num, 7))
            pos_params[0, :] = self.case_pos[0, 2:]
            pos_params[1:, :] = gen_param_results
            print('pos_params shape {}'.format(pos_params.shape))
            # time.sleep(30)
            return pos_params
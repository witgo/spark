/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.deploy.k8s.features

import java.util.UUID

import io.fabric8.kubernetes.api.model.{ContainerBuilder, HasMetadata, PersistentVolumeClaimBuilder, PersistentVolumeClaimVolumeSource, PodBuilder, Quantity, ResourceRequirementsBuilder, VolumeBuilder, VolumeMountBuilder}

import org.apache.spark.deploy.k8s.{KubernetesConf, KubernetesDriverSpecificConf, KubernetesExecutorSpecificConf, KubernetesRoleSpecificConf, SparkPod}

private[spark] class EvsDirsFeatureStep(
  conf: KubernetesConf[_ <: KubernetesRoleSpecificConf],
  defaultLocalDir: String = s"/var/data/spark-${UUID.randomUUID}")
  extends KubernetesFeatureConfigStep {

  // Cannot use Utils.getConfiguredLocalDirs because that will default to the Java system
  // property - we want to instead default to mounting an emptydir volume that doesn't already
  // exist in the image.
  // We could make utils.getConfiguredLocalDirs opinionated about Kubernetes, as it is already
  // a bit opinionated about YARN and Mesos.
  private val resolvedLocalDirs = Option(conf.sparkConf.getenv("SPARK_LOCAL_DIRS"))
    .orElse(conf.getOption("spark.local.dir"))
    .getOrElse(defaultLocalDir)
    .split(",")
  private val storageClass =
    conf.sparkConf.get("spark.kubernetes.cci.local.dir.storageClass", "ssd")
  private val quota =
    conf.sparkConf.get("spark.kubernetes.cci.local.dir.quota", "10Gi")

  def resourcesNamePrefix(index: Int): String = {
    conf.roleSpecificConf match {
      case _: KubernetesDriverSpecificConf =>
        s"${conf.appResourceNamePrefix}-driver-spark-local-dir-${index + 1}"
      case executorConf: KubernetesExecutorSpecificConf =>
        val executorId = executorConf.executorId
        s"${conf.appResourceNamePrefix}-executor-$executorId-spark-local-dir-${index + 1}"
    }
  }

  override def configurePod(pod: SparkPod): SparkPod = {
    val localDirVolumes = resolvedLocalDirs
      .zipWithIndex
      .map { case (localDir, index) =>
        new VolumeBuilder()
          .withName(s"spark-local-dir-${index + 1}")
          .withNewPersistentVolumeClaim()
          .withClaimName(resourcesNamePrefix(index))
          .withReadOnly(false)
          .endPersistentVolumeClaim()
          .build()
      }
    val localDirVolumeMounts = localDirVolumes
      .zip(resolvedLocalDirs)
      .map { case (localDirVolume, localDirPath) =>
        new VolumeMountBuilder()
          .withName(localDirVolume.getName)
          .withMountPath(localDirPath)
          .build()
      }
    val podWithLocalDirVolumes = new PodBuilder(pod.pod)
      .editSpec()
      .addToVolumes(localDirVolumes: _*)
      .endSpec()
      .build()
    val containerWithLocalDirVolumeMounts = new ContainerBuilder(pod.container)
      .addNewEnv()
      .withName("SPARK_LOCAL_DIRS")
      .withValue(resolvedLocalDirs.mkString(","))
      .endEnv()
      .addToVolumeMounts(localDirVolumeMounts: _*)
      .build()
    SparkPod(podWithLocalDirVolumes, containerWithLocalDirVolumeMounts)
  }

  override def getAdditionalPodSystemProperties(): Map[String, String] = Map.empty

  override def getAdditionalKubernetesResources(): Seq[HasMetadata] = {
    val storageClassKey = "volume.beta.kubernetes.io/storage-class"
    val storageProvisioner = "volume.beta.kubernetes.io/storage-provisioner"
    val localDirVolumes = resolvedLocalDirs
      .zipWithIndex
      .map { case (localDir, index) =>
        new PersistentVolumeClaimBuilder()
          .withNewMetadata()
          .withName(resourcesNamePrefix(index))
          .addToAnnotations(storageClassKey, storageClass)
          .addToAnnotations(storageProvisioner, "flexvolume-huawei.com/fuxivol")
          .endMetadata()
          .withNewSpec()
          .addToAccessModes("ReadWriteOnce")
          .editOrNewResources()
          .addToRequests("storage", new Quantity(quota))
          .endResources()
          .endSpec().build()
      }
    localDirVolumes
  }
}

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

package org.apache.spark.scheduler

import java.io._
import java.nio.ByteBuffer
import java.util.Properties

import scala.collection.JavaConverters._
import scala.collection.mutable

import org.apache.spark.serializer.SerializerInstance
import org.apache.spark.util.{ByteBufferInputStream, ByteBufferOutputStream}

/**
 * Description of a task that gets passed onto executors to be executed, usually created by
 * `TaskSetManager.resourceOffer`.
 */
private[spark] class TaskDescription private(
  val taskId: Long,
  val attemptNumber: Int,
  val executorId: String,
  val name: String,
  val index: Int,
  val taskFiles: mutable.Map[String, Long],
  val taskJars: mutable.Map[String, Long],
  private var task_ : Task[_],
  private var taskBytes: InputStream,
  private var taskProps: Properties) {

  def this(taskId: Long,
    attemptNumber: Int,
    executorId: String,
    name: String,
    index: Int,
    taskFiles: mutable.Map[String, Long],
    taskJars: mutable.Map[String, Long],
    task: Task[_]) {
    this(taskId, attemptNumber, executorId, name, index, taskFiles, taskJars, task,
      null.asInstanceOf[InputStream],
      null.asInstanceOf[Properties])
  }

  @throws[IOException]
  def encode(serializer: SerializerInstance): ByteBuffer = {
    val out = new ByteBufferOutputStream(4096)
    encode(out, serializer)
    out.close()
    out.toByteBuffer
  }

  @throws[IOException]
  def encode(outputStream: OutputStream, serializer: SerializerInstance): Unit = {
    val out = new DataOutputStream(outputStream)
    // Write taskId
    out.writeLong(taskId)

    // Write attemptNumber
    out.writeInt(attemptNumber)

    // Write executorId
    out.writeUTF(executorId)

    // Write name
    out.writeUTF(name)

    // Write index
    out.writeInt(index)

    // Write taskFiles
    out.writeInt(taskFiles.size)
    for ((name, timestamp) <- taskFiles) {
      out.writeUTF(name)
      out.writeLong(timestamp)
    }

    // Write taskJars
    out.writeInt(taskJars.size)
    for ((name, timestamp) <- taskJars) {
      out.writeUTF(name)
      out.writeLong(timestamp)
    }

    // Write the task properties separately so it is available before full task deserialization.
    val taskProps = task_.localProperties
    val propertyNames = taskProps.stringPropertyNames
    out.writeInt(propertyNames.size())
    propertyNames.asScala.foreach { key =>
      val value = taskProps.getProperty(key)
      out.writeUTF(key)
      out.writeUTF(value)
    }

    // Write the task itself and finish
    val serializeStream = serializer.serializeStream(out)
    serializeStream.writeValue(task_)
    serializeStream.flush()
  }

  def task(ser: SerializerInstance): Task[_] = {
    if (task_ == null) {
      val deserializeStream = ser.deserializeStream(taskBytes)
      task_ = deserializeStream.readValue[Task[_]]()
      task_.localProperties = taskProps
      deserializeStream.close()
      taskProps = null
      taskBytes = null
    }
    task_
  }

  def taskProperties(): Properties = {
    if (task_ != null) {
      task_.localProperties
    } else {
      taskProps
    }
  }

  override def toString: String = "TaskDescription(TID=%d, index=%d)".format(taskId, index)
}

private[spark] object TaskDescription {

  @throws[IOException]
  def apply(byteBuffer: ByteBuffer): TaskDescription = {
    decode(byteBuffer)
  }

  @throws[IOException]
  def decode(byteBuffer: ByteBuffer): TaskDescription = {
    decode(new ByteBufferInputStream(byteBuffer))
  }

  @throws[IOException]
  def decode(inputStream: InputStream): TaskDescription = {
    val in = new DataInputStream(inputStream)
    // Read taskId
    val taskId = in.readLong()

    // Read attemptNumber
    val attemptNumber = in.readInt()

    // Read executorId
    val executorId = in.readUTF()

    // Read name
    val name = in.readUTF()

    // Read index
    val index = in.readInt()

    // Read task's files
    val taskFiles = new mutable.HashMap[String, Long]()
    val numFiles = in.readInt()
    for (i <- 0 until numFiles) {
      taskFiles(in.readUTF()) = in.readLong()
    }

    // Read task's JARs
    val taskJars = new mutable.HashMap[String, Long]()
    val numJars = in.readInt()
    for (i <- 0 until numJars) {
      taskJars(in.readUTF()) = in.readLong()
    }

    // Read task's properties
    val taskProps = new Properties()
    val numProps = in.readInt()
    for (i <- 0 until numProps) {
      val key = in.readUTF()
      val value = in.readUTF()
      taskProps.setProperty(key, value)
    }

    new TaskDescription(taskId, attemptNumber, executorId, name, index,
      taskFiles, taskJars, null.asInstanceOf[Task[_]], in, taskProps)
  }
}

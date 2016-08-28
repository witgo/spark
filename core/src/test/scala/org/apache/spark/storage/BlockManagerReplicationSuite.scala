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

package org.apache.spark.storage

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration._
import scala.language.implicitConversions
import scala.language.postfixOps

import org.mockito.Mockito.{mock, when}
import org.scalatest.{BeforeAndAfter, Matchers}
import org.scalatest.concurrent.Eventually._

import org.apache.spark._
import org.apache.spark.broadcast.BroadcastManager
import org.apache.spark.memory.UnifiedMemoryManager
import org.apache.spark.network.BlockTransferService
import org.apache.spark.network.netty.NettyBlockTransferService
import org.apache.spark.rpc.RpcEnv
import org.apache.spark.scheduler.LiveListenerBus
import org.apache.spark.serializer.{KryoSerializer, SerializerManager}
import org.apache.spark.shuffle.sort.SortShuffleManager
import org.apache.spark.storage.StorageLevel._

/** Testsuite that tests block replication in BlockManager */
class BlockManagerReplicationSuite extends SparkFunSuite
    with Matchers
    with BeforeAndAfter
    with LocalSparkContext {

  private val conf = new SparkConf(false).set("spark.app.id", "test")
  private var rpcEnv: RpcEnv = null
  private var master: BlockManagerMaster = null
  private val securityMgr = new SecurityManager(conf)
  private val bcastManager = new BroadcastManager(true, conf, securityMgr)
  private val mapOutputTracker = new MapOutputTrackerMaster(conf, bcastManager, true)
  private val shuffleManager = new SortShuffleManager(conf)

  // List of block manager created during an unit test, so that all of the them can be stopped
  // after the unit test.
  private val allStores = new ArrayBuffer[BlockManager]

  // Reuse a serializer across tests to avoid creating a new thread-local buffer on each test
  conf.set("spark.kryoserializer.buffer", "1m")
  private val serializer = new KryoSerializer(conf)

  // Implicitly convert strings to BlockIds for test clarity.
  private implicit def StringToBlockId(value: String): BlockId = new TestBlockId(value)

  private def makeBlockManager(
      maxMem: Long,
      name: String = SparkContext.DRIVER_IDENTIFIER): BlockManager = {
    conf.set("spark.testing.memory", maxMem.toString)
    conf.set("spark.memory.offHeap.size", maxMem.toString)
    val transfer = new NettyBlockTransferService(conf, securityMgr, "localhost", numCores = 1)
    val memManager = UnifiedMemoryManager(conf, numCores = 1)
    val serializerManager = new SerializerManager(serializer, conf)
    val store = new BlockManager(name, rpcEnv, master, serializerManager, conf,
      memManager, mapOutputTracker, shuffleManager, transfer, securityMgr, 0)
    memManager.setMemoryStore(store.memoryStore)
    store.initialize("app-id")
    allStores += store
    store
  }

  before {
    rpcEnv = RpcEnv.create("test", "localhost", 0, conf, securityMgr)

    conf.set("spark.authenticate", "false")
    conf.set("spark.driver.port", rpcEnv.address.port.toString)
    conf.set("spark.testing", "true")
    conf.set("spark.memory.fraction", "1")
    conf.set("spark.memory.storageFraction", "1")
    conf.set("spark.storage.unrollFraction", "0.4")
    conf.set("spark.storage.unrollMemoryThreshold", "512")

    // to make a replication attempt to inactive store fail fast
    conf.set("spark.core.connection.ack.wait.timeout", "1s")
    // to make cached peers refresh frequently
    conf.set("spark.storage.cachedPeersTtl", "10")

    sc = new SparkContext("local", "test", conf)
    master = new BlockManagerMaster(rpcEnv.setupEndpoint("blockmanager",
      new BlockManagerMasterEndpoint(rpcEnv, true, conf,
        new LiveListenerBus(sc))), conf, true)
    allStores.clear()
  }

  after {
    allStores.foreach { _.stop() }
    allStores.clear()
    rpcEnv.shutdown()
    rpcEnv.awaitTermination()
    rpcEnv = null
    master = null
  }

  test("block replication - 2x replication") {
    testReplication(2,
      Seq(MEMORY_AND_DISK_SER_2)
    )
  }

  /**
   * Test replication of blocks with different storage levels (various combinations of
   * memory, disk & serialization). For each storage level, this function tests every store
   * whether the block is present and also tests the master whether its knowledge of blocks
   * is correct. Then it also drops the block from memory of each store (using LRU) and
   * again checks whether the master's knowledge gets updated.
   */
  private def testReplication(maxReplication: Int, storageLevels: Seq[StorageLevel]) {
    import org.apache.spark.storage.StorageLevel._

    assert(maxReplication > 1,
      s"Cannot test replication factor $maxReplication")

    // storage levels to test with the given replication factor

    val storeSize = 10000
    val blockSize = 1000

    // As many stores as the replication factor
    val stores = (1 to maxReplication).map {
      i => makeBlockManager(storeSize, s"store$i")
    }

    storageLevels.foreach { storageLevel =>
      // Put the block into one of the stores
      val blockId = new TestBlockId(
        "block-with-" + storageLevel.description.replace(" ", "-").toLowerCase)
      stores(0).putSingle(blockId, new Array[Byte](blockSize), storageLevel)

      // Assert that master know two locations for the block
      val blockLocations = master.getLocations(blockId).map(_.executorId).toSet
      assert(blockLocations.size === storageLevel.replication,
        s"master did not have ${storageLevel.replication} locations for $blockId")

      // Test state of the stores that contain the block
      stores.filter {
        testStore => blockLocations.contains(testStore.blockManagerId.executorId)
      }.foreach { testStore =>
        val testStoreName = testStore.blockManagerId.executorId
        assert(
          testStore.getLocalValues(blockId).isDefined, s"$blockId was not found in $testStoreName")
        testStore.releaseLock(blockId)
        assert(master.getLocations(blockId).map(_.executorId).toSet.contains(testStoreName),
          s"master does not have status for ${blockId.name} in $testStoreName")

        val blockStatus = master.getBlockStatus(blockId)(testStore.blockManagerId)

        // Assert that block status in the master for this store has expected storage level
        assert(
          blockStatus.storageLevel.useDisk === storageLevel.useDisk &&
            blockStatus.storageLevel.useMemory === storageLevel.useMemory &&
            blockStatus.storageLevel.useOffHeap === storageLevel.useOffHeap &&
            blockStatus.storageLevel.deserialized === storageLevel.deserialized,
          s"master does not know correct storage level for ${blockId.name} in $testStoreName")

        // Assert that the block status in the master for this store has correct memory usage info
        assert(!blockStatus.storageLevel.useMemory || blockStatus.memSize >= blockSize,
          s"master does not know size of ${blockId.name} stored in memory of $testStoreName")


        // If the block is supposed to be in memory, then drop the copy of the block in
        // this store test whether master is updated with zero memory usage this store
        if (storageLevel.useMemory) {
          val sl = if (storageLevel.useOffHeap) {
            StorageLevel(false, true, true, false, 1)
          } else {
            MEMORY_ONLY_SER
          }
          // Force the block to be dropped by adding a number of dummy blocks
          (1 to 10).foreach {
            i => testStore.putSingle(s"dummy-block-$i", new Array[Byte](1000), sl)
          }
          (1 to 10).foreach {
            i => testStore.removeBlock(s"dummy-block-$i")
          }

          val newBlockStatusOption = master.getBlockStatus(blockId).get(testStore.blockManagerId)

          // Assert that the block status in the master either does not exist (block removed
          // from every store) or has zero memory usage for this store
          assert(
            newBlockStatusOption.isEmpty || newBlockStatusOption.get.memSize === 0,
            s"after dropping, master does not know size of ${blockId.name} " +
              s"stored in memory of $testStoreName"
          )
        }

        // If the block is supposed to be in disk (after dropping or otherwise, then
        // test whether master has correct disk usage for this store
        if (storageLevel.useDisk) {
          assert(master.getBlockStatus(blockId)(testStore.blockManagerId).diskSize >= blockSize,
            s"after dropping, master does not know size of ${blockId.name} " +
              s"stored in disk of $testStoreName"
          )
        }
      }
      master.removeBlock(blockId)
    }
  }
}

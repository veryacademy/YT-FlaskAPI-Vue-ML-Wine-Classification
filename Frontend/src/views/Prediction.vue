<template>
  <div
    class="d-flex flex-column justify-content-center align-items-center"
    style="height: 80vh"
  >
    <div class="p-2">
      <h1 v-if="!APIResult.length">Enter Wine Data to make prediction</h1>
      <h1 v-else style="font-size:4rem">{{ APIResult }}</h1>
    </div>
    <div class="p-2">
      <form @submit.prevent>
        <div class="form-row" style="max-width:500px">
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="a">a</label>
            <input
              class="form-control"
              v-model.trim="winedata.a"
              type="number"
              id="a"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="b">b</label>
            <input
              class="form-control"
              v-model.trim="winedata.b"
              type="number"
              id="a"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="c">c</label>
            <input
              class="form-control"
              v-model.trim="winedata.c"
              type="number"
              id="c"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="d">d</label>
            <input
              class="form-control"
              v-model.trim="winedata.d"
              type="number"
              id="d"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="e">e</label>
            <input
              class="form-control"
              v-model.trim="winedata.e"
              type="number"
              id="e"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="f">f</label>
            <input
              class="form-control"
              v-model.trim="winedata.f"
              type="number"
              id="f"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="g">g</label>
            <input
              class="form-control"
              v-model.trim="winedata.g"
              type="number"
              id="g"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="h">h</label>
            <input
              class="form-control"
              v-model.trim="winedata.h"
              type="number"
              id="h"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="i">i</label>
            <input
              class="form-control"
              v-model.trim="winedata.i"
              type="number"
              id="i"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="j">j</label>
            <input
              class="form-control"
              v-model.trim="winedata.j"
              type="number"
              id="j"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="k">k</label>
            <input
              class="form-control"
              v-model.trim="winedata.k"
              type="number"
              id="k"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="l">l</label>
            <input
              class="form-control"
              v-model.trim="winedata.l"
              type="number"
              id="l"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="m">m</label>
            <input
              class="form-control"
              v-model.trim="winedata.m"
              type="number"
              id="m"
            />
          </div>
          <button
            @click="predict"
            type="button"
            class="btn btn-primary btn-lg btn-block"
          >
            Submit
          </button>
        </div>
      </form>
    </div>
  </div>
</template>

<script>
import { getAPI } from "@/axios";
export default {
  name: "Posts",
  data() {
    return {
      winedata: {
        a: "",
        b: "",
        c: "",
        d: "",
        e: "",
        f: "",
        g: "",
        h: "",
        i: "",
        j: "",
        k: "",
        l: ""
      },
      APIResult: []
    };
  },
  methods: {
    predict() {
      getAPI
        .get("/wine", {
          params: {
            alcohol: this.winedata.a,
            ash: this.winedata.b,
            alcalinity_of_ash: this.winedata.c,
            magnesium: this.winedata.d,
            total_phenols: this.winedata.e,
            flavanoids: this.winedata.f,
            nonflavanoid_phenols: this.winedata.g,
            proanthocyanins: this.winedata.h,
            color_intensity: this.winedata.i,
            hue: this.winedata.j,
            od315_of_diluted_wines: this.winedata.k,
            proline: this.winedata.l,
            malic_acid: this.winedata.m
          }
        })
        .then(response => {
          console.log("Recieved data successfully");
          this.APIResult = response.data;
          console.log(response.data);
        })
        .catch(err => {
          console.log(err);
        });
    }
  }
};
</script>

<style>
</style>


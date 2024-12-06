<template>
  <div class="bt-loading" v-if="active">
    <cv-loading :active="active" :overlay="false" ></cv-loading>
    <label> {{text}}</label>
</div>
</template>

<script>
export default {
name:"LoadingDialog",
props:['text','active']
}
</script>

<style lang="scss">
.bt-loading{
    position: absolute;
    top: 50%;
    left: 50%;
    margin-right: -50%;
    transform: translate(-50%, -50%);
  }
</style>
<script>
export default {
  props: ['message', 'highlight', 'highlightmsg', 'probability'],
  computed: {
    splittedMsg() {
      return this.message.toLowerCase().split(this.highlight)
    }
  }
}
</script>

<template>
  <main>
    <div v-if="probability.includes('high')" >
      {{ splittedMsg[0] }} 
      <span class="tooltipWrapper red">{{ highlight }}
        <span class="tooltip">{{ highlightmsg }}
        </span>
        <!-- <button class="bx--button">Accept Suggestion</button> -->
      </span>{{ splittedMsg[1] }}
    </div>
    <div v-if="probability.includes('medium')" >
      {{ splittedMsg[0] }} <span class="tooltipWrapper yellow">{{ highlight }}<span class="tooltip">{{ highlightmsg }}</span></span>{{ splittedMsg[1] }}
    </div>
    <div v-if="probability.includes('low')" >
      {{ this.message }}
    </div>
  </main>
</template>

<style scoped>
.red {
  background: red;
}

.yellow {
  background: yellow;
}

.tooltipWrapper {
  position: relative;
}

.tooltip {
  display: none;
  position: relative;
  top: -40px;
  left: 500%;
  transform: translateX(-50%);
  color: white;
  background: #161616;
  white-space: pre;
  padding: 2px 8px;
}

.tooltipWrapper:hover .tooltip {
  display: block;
  position: absolute;
}
</style>
